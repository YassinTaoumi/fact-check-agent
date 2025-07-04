package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"go.mau.fi/whatsmeow"
	waProto "go.mau.fi/whatsmeow/binary/proto"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
	
	_ "github.com/mattn/go-sqlite3" // SQLite driver
	"github.com/mdp/qrterminal"
)

var globalClient *whatsmeow.Client

// SendMessageRequest represents the request to send a WhatsApp message
type SendMessageRequest struct {
	Recipient string `json:"recipient"` // Can be JID or phone number
	Message   string `json:"message"`
}

// SendMessageResponse represents the response from sending a message
type SendMessageResponse struct {
	Success   bool   `json:"success"`
	MessageID string `json:"message_id,omitempty"`
	Error     string `json:"error,omitempty"`
}

// parseRecipient converts phone number or JID to proper WhatsApp JID
func parseRecipient(recipient string) (types.JID, error) {
	// If it's already a JID, parse it
	if strings.Contains(recipient, "@") {
		jid, err := types.ParseJID(recipient)
		if err != nil {
			return types.EmptyJID, fmt.Errorf("invalid JID format: %s", recipient)
		}
		return jid, nil
	}
	
	// If it's a phone number, convert to WhatsApp JID
	// Remove any non-digit characters
	phoneNumber := strings.ReplaceAll(recipient, "+", "")
	phoneNumber = strings.ReplaceAll(phoneNumber, "-", "")
	phoneNumber = strings.ReplaceAll(phoneNumber, " ", "")
	
	if len(phoneNumber) < 10 {
		return types.EmptyJID, fmt.Errorf("phone number too short: %s", phoneNumber)
	}
	
	// Create WhatsApp JID for individual chat
	jid := types.NewJID(phoneNumber, types.DefaultUserServer)
	return jid, nil
}

// sendWhatsAppMessage sends a message via WhatsApp
func sendWhatsAppMessage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   "Only POST method allowed",
		})
		return
	}
	
	// Check if client is connected
	if globalClient == nil || !globalClient.IsConnected() {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   "WhatsApp client not connected",
		})
		return
	}
	
	// Parse request
	var req SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   "Invalid JSON format",
		})
		return
	}
	
	// Validate request
	if req.Recipient == "" || req.Message == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   "Both recipient and message are required",
		})
		return
	}
	
	// Parse recipient
	recipientJID, err := parseRecipient(req.Recipient)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   fmt.Sprintf("Invalid recipient: %s", err.Error()),
		})
		return
	}
	
	// Create message
	message := &waProto.Message{
		Conversation: &req.Message,
	}
	
	// Send message
	resp, err := globalClient.SendMessage(context.Background(), recipientJID, message)
	if err != nil {
		log.Printf("Failed to send message to %s: %v", recipientJID, err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(SendMessageResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to send message: %s", err.Error()),
		})
		return
	}
	
	log.Printf("✅ Message sent to %s, ID: %s", recipientJID, resp.ID)
	
	// Success response
	json.NewEncoder(w).Encode(SendMessageResponse{
		Success:   true,
		MessageID: resp.ID,
	})
}

// pairDevice provides pairing status information (pairing happens automatically on startup)
func pairDevice(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	if globalClient == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": false,
			"error":   "WhatsApp client not initialized",
		})
		return
	}
	
	// Check current pairing and connection status
	paired := globalClient.Store != nil && globalClient.Store.ID != nil
	connected := globalClient.IsConnected()
	
	if paired && connected {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":   true,
			"message":   "Device is paired and connected",
			"paired":    true,
			"connected": true,
		})
		return
	}
	
	if paired && !connected {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success":   false,
			"message":   "Device is paired but not connected. Try restarting the server.",
			"paired":    true,
			"connected": false,
		})
		return
	}
	
	// Not paired
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": false,
		"message": "Device not paired. Restart the server to see QR code in console.",
		"paired":  false,
		"connected": false,
		"instructions": []string{
			"1. Restart the WhatsApp API server",
			"2. Check the server console for QR code",
			"3. Open WhatsApp on your phone",
			"4. Go to Settings > Linked Devices",
			"5. Tap 'Link a Device'",
			"6. Scan the QR code displayed in the server console",
		},
	})
}

// healthCheck provides a health check endpoint
func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	connected := globalClient != nil && globalClient.IsConnected()
	paired := globalClient != nil && globalClient.Store != nil && globalClient.Store.ID != nil
	
	status := map[string]interface{}{
		"status":             "ok",
		"whatsapp_connected": connected,
		"device_paired":      paired,
		"timestamp":          time.Now().UTC().Format(time.RFC3339),
	}
	
	if connected && paired {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
		status["status"] = "degraded"
		if !paired {
			status["message"] = "Device not paired. Use /pair endpoint for pairing instructions."
		}
	}
	
	json.NewEncoder(w).Encode(status)
}

// initializeWhatsAppClient initializes the WhatsApp client and handles pairing automatically
func initializeWhatsAppClient() error {
	logger := waLog.Stdout("WhatsAppAPI", "INFO", true)
	
	// Create store directory if it doesn't exist (same as main.go)
	if err := os.MkdirAll("store", 0755); err != nil {
		return fmt.Errorf("failed to create store directory: %v", err)
	}
	
	// Initialize database with the same path as main.go
	container, err := sqlstore.New(context.Background(), "sqlite3", "file:store/whatsapp.db?_foreign_keys=on", logger)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %v", err)
	}
	
	deviceStore := container.NewDevice()
	
	// Create client
	globalClient = whatsmeow.NewClient(deviceStore, logger)
	if globalClient == nil {
		return fmt.Errorf("failed to create WhatsApp client")
	}
	
	// Add event handlers for connection status
	globalClient.AddEventHandler(func(evt interface{}) {
		switch evt.(type) {
		case *events.Connected:
			log.Println("✅ Connected to WhatsApp successfully!")
		case *events.LoggedOut:
			log.Println("⚠️  Device logged out - will need to restart server for new QR code")
		case *events.Disconnected:
			log.Println("📴 Disconnected from WhatsApp")
		}
	})
	
	// Check if device is already paired
	if deviceStore.ID == nil {
		log.Println("📱 Device not paired - generating QR code for pairing...")
		log.Println("🔳 QR Code for WhatsApp pairing:")
		log.Println("   1. Open WhatsApp on your phone")
		log.Println("   2. Go to Settings > Linked Devices")
		log.Println("   3. Tap 'Link a Device'")
		log.Println("   4. Scan the QR code below:")
		log.Println("")
		
		// Start pairing process with QR code immediately
		qrChan, _ := globalClient.GetQRChannel(context.Background())
		err := globalClient.Connect()
		if err != nil {
			return fmt.Errorf("failed to start pairing process: %v", err)
		}
		
		// Handle QR code events
		for evt := range qrChan {
			if evt.Event == "code" {
				log.Println("� QR Code:")
				qrterminal.GenerateHalfBlock(evt.Code, qrterminal.L, os.Stdout)
				log.Println("⏳ Waiting for QR code scan...")
			} else if evt.Event == "success" {
				log.Println("✅ QR code scanned successfully!")
				log.Println("🔄 Stabilizing connection...")
				time.Sleep(3 * time.Second)
				
				if globalClient.IsConnected() {
					log.Println("✅ WhatsApp client paired and connected!")
				} else {
					log.Println("⚠️ Connection not stable after pairing")
				}
				break
			} else if evt.Event == "timeout" {
				log.Println("⏰ QR code expired")
				return fmt.Errorf("QR code pairing timed out - please restart the server")
			}
		}
		
		return nil
	}
	
	// Device is already paired, try to connect
	log.Println("📱 Device already paired, connecting...")
	err = globalClient.Connect()
	if err != nil {
		log.Printf("⚠️ Failed to connect to WhatsApp (may need re-pairing): %v", err)
		return fmt.Errorf("failed to connect with existing pairing: %v", err)
	}
	
	// Wait a moment for connection to stabilize
	time.Sleep(2 * time.Second)
	
	if !globalClient.IsConnected() {
		return fmt.Errorf("WhatsApp client not connected after pairing")
	}
	
	log.Println("✅ WhatsApp client initialized and connected")
	return nil
}

func main() {
	log.Println("🚀 Starting WhatsApp API server...")
	
	// Initialize WhatsApp client (with automatic QR code pairing if needed)
	if err := initializeWhatsAppClient(); err != nil {
		log.Printf("❌ Failed to initialize WhatsApp client: %v", err)
		log.Println("⚠️  WhatsApp API will be unavailable")
		log.Println("💡 Try restarting the server to attempt pairing again")
		// Don't exit - still provide API for status checking
	} else {
		log.Println("✅ WhatsApp client ready for sending messages")
	}
	
	// Setup HTTP routes
	http.HandleFunc("/send-message", sendWhatsAppMessage)
	http.HandleFunc("/health", healthCheck)
	http.HandleFunc("/pair", pairDevice)
	
	// Start HTTP server
	port := "9090"
	log.Printf("🌐 WhatsApp API server running on port %s", port)
	log.Printf("📤 Send messages: http://localhost:%s/send-message", port)
	log.Printf("🔍 Health check: http://localhost:%s/health", port)
	log.Printf("📱 Pairing status: http://localhost:%s/pair", port)
	log.Println("📋 Note: QR code pairing happens automatically on startup if device is not paired")
	
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Printf("❌ Server failed to start: %v", err)
	}
}
