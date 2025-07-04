package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/mdp/qrterminal"
	"go.mau.fi/whatsmeow"
	waProto "go.mau.fi/whatsmeow/binary/proto"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"
)

// API Configuration
const (
	ORCHESTRATION_API_URL = "http://localhost:8000"
	PROCESSING_API_URL    = "http://localhost:8001" // This is the main API we're using
	STORAGE_API_URL       = "http://localhost:8002"
	MAX_RETRIES           = 3
	RETRY_DELAY           = 5 * time.Second
	API_TIMEOUT           = 30 * time.Second
	FILE_STORAGE_BASE     = "file_storage" // Base directory for file storage
)

// API Request/Response Models
type ProcessingRequest struct {
	MessageID      string `json:"message_id"`
	ChatJID        string `json:"chat_jid"`
	ChatName       string `json:"chat_name"`
	SenderJID      string `json:"sender_jid"`
	SenderName     string `json:"sender_name"`
	UserIdentifier string `json:"user_identifier"`
	Content        string `json:"content"`
	ContentType    string `json:"content_type"`
	MediaFilename  string `json:"media_filename,omitempty"`
	MediaSize      int64  `json:"media_size,omitempty"`
	MediaPath      string `json:"media_path,omitempty"`
	IsFromMe       bool   `json:"is_from_me"`
	IsGroup        bool   `json:"is_group"`
	Timestamp      string `json:"timestamp"`
	SourceType     string `json:"source_type"`
	Priority       string `json:"priority"`
}

type ProcessingResponse struct {
	Success          bool                   `json:"success"`
	Message          string                 `json:"message"`
	StoredID         string                 `json:"stored_id,omitempty"`
	FileStoragePath  string                 `json:"file_storage_path,omitempty"`
	ProcessingJobID  string                 `json:"processing_job_id,omitempty"`
	ValidationResult map[string]interface{} `json:"validation_result,omitempty"`
	Error            string                 `json:"error,omitempty"`
	ShouldRetry      bool                   `json:"should_retry,omitempty"`
}

type HealthStatus struct {
	Status        string                 `json:"status"`
	Timestamp     string                 `json:"timestamp"`
	Services      map[string]interface{} `json:"services"`
	OverallHealth bool                   `json:"overall_health"`
}

type MessageContext struct {
	IsGroup     bool   `json:"is_group"`
	GroupName   string `json:"group_name,omitempty"`
	IsChannel   bool   `json:"is_channel"`
	ChannelName string `json:"channel_name,omitempty"`
}

// MediaDownloader implements the whatsmeow.DownloadableMessage interface
type MediaDownloader struct {
	URL           string
	DirectPath    string
	MediaKey      []byte
	FileLength    uint64
	FileSHA256    []byte
	FileEncSHA256 []byte
	MediaType     whatsmeow.MediaType
}

func (d *MediaDownloader) GetDirectPath() string             { return d.DirectPath }
func (d *MediaDownloader) GetURL() string                    { return d.URL }
func (d *MediaDownloader) GetMediaKey() []byte               { return d.MediaKey }
func (d *MediaDownloader) GetFileLength() uint64             { return d.FileLength }
func (d *MediaDownloader) GetFileSHA256() []byte             { return d.FileSHA256 }
func (d *MediaDownloader) GetFileEncSHA256() []byte          { return d.FileEncSHA256 }
func (d *MediaDownloader) GetMediaType() whatsmeow.MediaType { return d.MediaType }

// HTTP Client with timeout
var httpClient = &http.Client{
	Timeout: API_TIMEOUT,
}

// Link detection patterns
var (
	// URL regex pattern that matches http(s) URLs
	urlRegex = regexp.MustCompile(`https?://[^\s<>"{}|\\^` + "`" + `\[\]]+`)

	// Common domain patterns without protocol
	domainRegex = regexp.MustCompile(`(?i)\b(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:\.[a-z]{2,})?(?:/[^\s]*)?`)

	// Social media and messaging app links
	socialMediaRegex = regexp.MustCompile(`(?i)\b(?:instagram\.com|facebook\.com|twitter\.com|x\.com|linkedin\.com|tiktok\.com|youtube\.com|youtu\.be|whatsapp\.com|telegram\.me|t\.me)/[^\s]*`)
)

// Message Context Functions
func determineMessageContext(chatJID, chatName string, isGroup bool) MessageContext {
	context := MessageContext{
		IsGroup:   false,
		IsChannel: false,
	}

	if !isGroup || chatJID == "" || chatName == "" {
		return context
	}

	// Check if it's a WhatsApp channel
	if strings.Contains(chatJID, "@newsletter") {
		context.IsChannel = true
		context.ChannelName = chatName
		fmt.Printf("🔔 Detected WhatsApp channel: %s\n", chatName)
	} else if strings.Contains(chatJID, "@g.us") {
		// Regular WhatsApp group
		context.IsGroup = true
		context.GroupName = chatName
		fmt.Printf("👥 Detected WhatsApp group: %s\n", chatName)
	}

	return context
}

func formatContextMessage(context MessageContext, baseMessage string) string {
	if context.IsGroup {
		return fmt.Sprintf("%s from group: %s", baseMessage, context.GroupName)
	} else if context.IsChannel {
		return fmt.Sprintf("%s from channel: %s", baseMessage, context.ChannelName)
	}
	return baseMessage
}

// Link detection functions
func containsLink(text string) bool {
	if text == "" {
		return false
	}

	// Check for HTTP(S) URLs
	if urlRegex.MatchString(text) {
		return true
	}

	// Check for social media links
	if socialMediaRegex.MatchString(text) {
		return true
	}

	// Check for domain patterns (like google.com, example.org)
	matches := domainRegex.FindAllString(text, -1)
	for _, match := range matches {
		// Validate if it's a proper domain by checking if it can be parsed as URL
		if isValidDomain(match) {
			return true
		}
	}

	return false
}

func isValidDomain(domain string) bool {
	// Add protocol if missing for URL parsing
	testURL := domain
	if !strings.HasPrefix(strings.ToLower(domain), "http") {
		testURL = "http://" + domain
	}

	// Try to parse as URL
	parsedURL, err := url.Parse(testURL)
	if err != nil {
		return false
	}

	// Check if hostname is valid
	hostname := parsedURL.Hostname()
	if hostname == "" {
		return false
	}

	// Must contain at least one dot and valid TLD
	parts := strings.Split(hostname, ".")
	if len(parts) < 2 {
		return false
	}

	// Last part should be a valid TLD (at least 2 characters)
	tld := parts[len(parts)-1]
	if len(tld) < 2 || !regexp.MustCompile(`^[a-zA-Z]+$`).MatchString(tld) {
		return false
	}

	return true
}

func extractLinks(text string) []string {
	var links []string

	// Extract HTTP(S) URLs
	httpLinks := urlRegex.FindAllString(text, -1)
	links = append(links, httpLinks...)

	// Extract social media links
	socialLinks := socialMediaRegex.FindAllString(text, -1)
	links = append(links, socialLinks...)

	// Extract domain patterns
	domainMatches := domainRegex.FindAllString(text, -1)
	for _, match := range domainMatches {
		if isValidDomain(match) {
			// Skip if already found as HTTP URL
			found := false
			for _, existing := range links {
				if strings.Contains(existing, match) {
					found = true
					break
				}
			}
			if !found {
				links = append(links, match)
			}
		}
	}

	return links
}

// API helper functions
func makeAPIRequest(method, url string, data interface{}) (*http.Response, error) {
	var body io.Reader
	var contentType string

	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %v", err)
		}
		body = bytes.NewBuffer(jsonData)
		contentType = "application/json"
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	return httpClient.Do(req)
}

func checkAPIHealth() error {
	fmt.Println("🔍 Checking Processing API health...")

	// Check processing API (the main one we're using)
	resp, err := makeAPIRequest("GET", PROCESSING_API_URL+"/api/health", nil)
	if err != nil {
		return fmt.Errorf("processing API health check failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("processing API returned status %d", resp.StatusCode)
	}

	fmt.Println("✅ Processing API is healthy")
	return nil
}

// sendTextToOrchestrationAPI sends text message to the orchestration API
func sendTextToOrchestrationAPI(req ProcessingRequest, logger waLog.Logger) (*ProcessingResponse, error) {
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	fmt.Printf("🔍 DEBUG: JSON being sent to API: %s\n", string(jsonData))
	var lastErr error
	for attempt := 1; attempt <= MAX_RETRIES; attempt++ {
		resp, err := http.Post(PROCESSING_API_URL+"/api/process-message", "application/json", bytes.NewBuffer(jsonData))
		if err != nil {
			lastErr = err
			logger.Warnf("Processing API attempt %d failed: %v", attempt, err)
			if attempt < MAX_RETRIES {
				time.Sleep(RETRY_DELAY)
				continue
			}
			break
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = err
			continue
		}

		fmt.Printf("🔍 DEBUG: API Response Status: %d\n", resp.StatusCode)
		fmt.Printf("🔍 DEBUG: API Response Body: %s\n", string(body))

		var processingResp ProcessingResponse
		if err := json.Unmarshal(body, &processingResp); err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
			return &processingResp, nil
		}

		lastErr = fmt.Errorf("API returned status %d: %s", resp.StatusCode, processingResp.Error)

		// Only retry if the API says we should
		if !processingResp.ShouldRetry || attempt == MAX_RETRIES {
			break
		}

		logger.Warnf("Processing failed (attempt %d), retrying: %s", attempt, processingResp.Error)
		time.Sleep(RETRY_DELAY)
	}

	return nil, fmt.Errorf("processing failed after %d attempts: %v", MAX_RETRIES, lastErr)
}

// sendFileToOrchestrationAPI sends file message to the orchestration API
func sendFileToOrchestrationAPI(req ProcessingRequest, filePath string, logger waLog.Logger) (*ProcessingResponse, error) {
	var lastErr error
	for attempt := 1; attempt <= MAX_RETRIES; attempt++ {
		// Create new buffer for each attempt
		var buf bytes.Buffer
		writer := multipart.NewWriter(&buf)

		// Add file
		if filePath != "" {
			file, err := os.Open(filePath)
			if err != nil {
				return nil, fmt.Errorf("failed to open file: %v", err)
			}
			defer file.Close()

			fileWriter, err := writer.CreateFormFile("file", req.MediaFilename)
			if err != nil {
				return nil, fmt.Errorf("failed to create form file: %v", err)
			}

			if _, err := io.Copy(fileWriter, file); err != nil {
				return nil, fmt.Errorf("failed to copy file: %v", err)
			}
		}

		// Add form fields - ensure all values are strings
		fields := map[string]string{
			"user_identifier": req.UserIdentifier,
			"message_id":      req.MessageID,
			"chat_jid":        req.ChatJID,
			"chat_name":       req.ChatName,
			"sender_jid":      req.SenderJID,
			"sender_name":     req.SenderName,
			"timestamp":       req.Timestamp,
			"content":         req.Content,
			"content_type":    req.ContentType,
			"media_filename":  req.MediaFilename,
			"media_size":      fmt.Sprintf("%d", req.MediaSize),
			"is_from_me":      fmt.Sprintf("%t", req.IsFromMe),
			"is_group":        fmt.Sprintf("%t", req.IsGroup),
			"source_type":     req.SourceType,
			"priority":        req.Priority,
		}
		fmt.Printf("🔍 DEBUG: Fields being sent to API: %+v\n", fields)
		for key, value := range fields {
			if err := writer.WriteField(key, value); err != nil {
				return nil, fmt.Errorf("failed to write field %s: %v", key, err)
			}
		}

		if err := writer.Close(); err != nil {
			return nil, fmt.Errorf("failed to close writer: %v", err)
		}
		// Create request
		httpReq, err := http.NewRequest("POST", PROCESSING_API_URL+"/api/process-file", &buf)
		if err != nil {
			lastErr = err
			logger.Errorf("Failed to create request: %v", err)
			fmt.Printf("❌ Failed to create request: %v\n", err)
			continue
		}

		httpReq.Header.Set("Content-Type", writer.FormDataContentType())
		fmt.Printf("📤 Sending file to %s, size: %d bytes, content-type: %s\n",
			PROCESSING_API_URL+"/api/process-file", buf.Len(), writer.FormDataContentType())

		resp, err := httpClient.Do(httpReq)
		if err != nil {
			lastErr = err
			logger.Warnf("File processing attempt %d failed: %v", attempt, err)
			fmt.Printf("❌ API request failed: %v\n", err)
			if attempt < MAX_RETRIES {
				time.Sleep(RETRY_DELAY)
				continue
			}
			break
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			lastErr = err
			logger.Errorf("Failed to read response body: %v", err)
			fmt.Printf("❌ Failed to read response body: %v\n", err)
			continue
		}

		fmt.Printf("📥 API response status: %d\n", resp.StatusCode)
		fmt.Printf("📄 Response body (first 200 chars): %s\n", string(body)[:min(200, len(string(body)))])

		var processingResp ProcessingResponse
		if err := json.Unmarshal(body, &processingResp); err != nil {
			lastErr = err
			logger.Errorf("Failed to unmarshal response: %v", err)
			fmt.Printf("❌ Failed to unmarshal response: %v\n", err)
			continue
		}

		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusCreated {
			return &processingResp, nil
		}

		lastErr = fmt.Errorf("API returned status %d: %s", resp.StatusCode, processingResp.Error)

		if !processingResp.ShouldRetry || attempt == MAX_RETRIES {
			break
		}

		logger.Warnf("File processing failed (attempt %d), retrying: %s", attempt, processingResp.Error)
		time.Sleep(RETRY_DELAY)
	}

	return nil, fmt.Errorf("file processing failed after %d attempts: %v", MAX_RETRIES, lastErr)
}

// ensureFileStorageDirectory creates the file_storage directory structure
func ensureFileStorageDirectory() error {
	if err := os.MkdirAll(FILE_STORAGE_BASE, 0755); err != nil {
		return fmt.Errorf("failed to create file_storage directory: %v", err)
	}
	return nil
}

// getUserFileStorageDir returns the directory path for a user's files
func getUserFileStorageDir(phoneNumber string) string {
	return filepath.Join(FILE_STORAGE_BASE, phoneNumber)
}

// downloadMediaFile downloads and saves media directly to file_storage
func downloadMediaFile(client *whatsmeow.Client, chatJID, senderJID, filename, url string,
	mediaKey, fileSHA256, fileEncSHA256 []byte, fileLength uint64, mediaType string) (string, int64, error) {

	if url == "" || len(mediaKey) == 0 {
		return "", 0, fmt.Errorf("insufficient media information for download")
	}

	// Extract phone number from sender JID
	phoneNumber := strings.Split(senderJID, "@")[0]

	// Create directory structure in file_storage
	userDir := getUserFileStorageDir(phoneNumber)
	if err := os.MkdirAll(userDir, 0755); err != nil {
		return "", 0, fmt.Errorf("failed to create user directory %s: %v", userDir, err)
	}

	// Generate unique filename with timestamp to avoid conflicts
	timestamp := time.Now().Format("20060102_150405")
	name := strings.TrimSuffix(filename, filepath.Ext(filename))
	ext := filepath.Ext(filename)
	uniqueFilename := fmt.Sprintf("%s_%s%s", timestamp, name, ext)

	// Create full file path in file_storage
	localPath := filepath.Join(userDir, uniqueFilename)

	fmt.Printf("⏬ Beginning media download process: %s\n", filename)
	fmt.Printf("    URL: %s\n", url)
	fmt.Printf("    Sender: %s\n", senderJID)
	fmt.Printf("    Media Type: %s\n", mediaType)
	fmt.Printf("    Storage Directory: %s\n", userDir)
	fmt.Printf("    Final Path: %s\n", localPath)

	// Check if file already exists (unlikely with timestamp, but just in case)
	if info, err := os.Stat(localPath); err == nil {
		fmt.Printf("📁 Media already exists: %s (%d bytes)\n", localPath, info.Size())
		return localPath, info.Size(), nil
	}

	// Extract direct path from URL
	directPath := extractDirectPathFromURL(url)

	// Determine WhatsApp media type
	var waMediaType whatsmeow.MediaType
	switch mediaType {
	case "image":
		waMediaType = whatsmeow.MediaImage
	case "video":
		waMediaType = whatsmeow.MediaVideo
	case "audio":
		waMediaType = whatsmeow.MediaAudio
	case "document":
		waMediaType = whatsmeow.MediaDocument
	default:
		waMediaType = whatsmeow.MediaDocument
	}

	// Create downloader
	downloader := &MediaDownloader{
		URL:           url,
		DirectPath:    directPath,
		MediaKey:      mediaKey,
		FileLength:    fileLength,
		FileSHA256:    fileSHA256,
		FileEncSHA256: fileEncSHA256,
		MediaType:     waMediaType,
	}

	fmt.Printf("⬇️  Downloading %s: %s\n", mediaType, filename)

	// Download the media
	mediaData, err := client.Download(context.Background(), downloader)
	if err != nil {
		return "", 0, fmt.Errorf("failed to download media: %v", err)
	}

	// Save to file
	if err := os.WriteFile(localPath, mediaData, 0644); err != nil {
		return "", 0, fmt.Errorf("failed to save media file: %v", err)
	}

	fmt.Printf("✅ Downloaded and saved: %s (%d bytes)\n", localPath, len(mediaData))
	return localPath, int64(len(mediaData)), nil
}

// Extract direct path from WhatsApp media URL
func extractDirectPathFromURL(url string) string {
	parts := strings.SplitN(url, ".net/", 2)
	if len(parts) < 2 {
		return url
	}
	pathPart := strings.SplitN(parts[1], "?", 2)[0]
	return "/" + pathPart
}

func handleIncomingMessage(client *whatsmeow.Client, msg *events.Message, logger waLog.Logger) {
	// ONLY process incoming messages (not sent by me)
	if msg.Info.IsFromMe {
		return
	}

	chatJID := msg.Info.Chat.String()
	senderJID := msg.Info.Sender.String()
	messageID := msg.Info.ID
	timestamp := msg.Info.Timestamp
	isFromMe := msg.Info.IsFromMe
	isGroup := strings.HasSuffix(chatJID, "@g.us") || strings.HasSuffix(chatJID, "@newsletter")

	chatName := getChatName(client, msg.Info.Chat, isGroup)
	senderName := getSenderName(client, msg.Info.Sender)
	content := extractTextContent(msg.Message)
	mediaType, filename, url, mediaKey, fileSHA256, fileEncSHA256, fileLength := extractMediaInfo(msg.Message)

	// Determine message context (group/channel)
	messageContext := determineMessageContext(chatJID, chatName, isGroup)

	// Debug: Log what we extracted
	if mediaType != "" {
		fmt.Printf("🎬 Media detected: type=%s, filename=%s, url_present=%v, key_present=%v\n",
			mediaType, filename, url != "", len(mediaKey) > 0)
	}

	// Skip if there's no content and no media
	if content == "" && mediaType == "" {
		return
	}

	var localMediaPath string
	var mediaSize int64 // Extract just the phone number from sender JID for validation (remove @s.whatsapp.net)
	phoneNumber := ""
	if senderJID != "" && strings.Contains(senderJID, "@") {
		phoneNumber = strings.Split(senderJID, "@")[0]
	}
	if phoneNumber == "" {
		logger.Errorf("[FATAL] Could not extract sender phone number from senderJID: %s. Skipping message to avoid DB error.", senderJID)
		return
	}

	// Download media if present
	if mediaType != "" && url != "" {
		var err error
		fmt.Printf("📥 Downloading %s media: %s\n", mediaType, filename)
		localMediaPath, mediaSize, err = downloadMediaFile(client, chatJID, senderJID, filename, url,
			mediaKey, fileSHA256, fileEncSHA256, fileLength, mediaType)
		if err != nil {
			logger.Errorf("Failed to download media: %v", err)
			fmt.Printf("❌ Media download failed: %v\n", err)
			// Continue processing the message even if media download fails
		} else {
			fmt.Printf("✅ Media downloaded successfully to file_storage: %s (%d bytes)\n", localMediaPath, mediaSize)
		}
	}

	// Determine content type with link detection
	contentType := "text"
	if mediaType != "" {
		switch mediaType {
		case "image":
			contentType = "image"
		case "video":
			contentType = "video"
		case "audio":
			contentType = "audio"
		case "document":
			if strings.Contains(strings.ToLower(filename), ".pdf") {
				contentType = "pdf"
			} else {
				contentType = "document"
			}
		default:
			contentType = "document"
		}
	} else if content != "" {
		// Check if text content contains links
		if containsLink(content) {
			contentType = "link"
			// Extract and log the links found
			links := extractLinks(content)
			fmt.Printf("🔗 Link(s) detected in message: %v\n", links)
		} else {
			contentType = "text"
		}
	} // Create processing request with enhanced context
	processingReq := ProcessingRequest{
		MessageID:      messageID,
		ChatJID:        chatJID,
		ChatName:       chatName,
		SenderJID:      senderJID,
		SenderName:     senderName,
		UserIdentifier: senderJID, // Use full sender JID instead of just phone number
		Content:        content,
		ContentType:    contentType,
		MediaFilename:  filename,
		MediaSize:      mediaSize,
		MediaPath:      localMediaPath, // This is the file_storage path
		IsFromMe:       isFromMe,
		IsGroup:        isGroup,
		Timestamp:      timestamp.Format(time.RFC3339),
		SourceType:     "whatsapp",
		Priority:       "normal",
	}

	// DEBUG: Log what we're sending
	fmt.Printf("🐛 DEBUG: Processing request data:\n")
	fmt.Printf("  MessageID: '%s'\n", processingReq.MessageID)
	fmt.Printf("  ChatJID: '%s'\n", processingReq.ChatJID)
	fmt.Printf("  SenderJID: '%s'\n", processingReq.SenderJID)
	fmt.Printf("  UserIdentifier: '%s'\n", processingReq.UserIdentifier)
	fmt.Printf("  Content: '%s'\n", processingReq.Content)
	fmt.Printf("  ContentType: '%s'\n", processingReq.ContentType)

	var response *ProcessingResponse
	var err error
	if mediaType != "" && localMediaPath != "" {
		// Send file with the file_storage path
		fmt.Printf("📤 Sending file to API: %s\n", localMediaPath)
		response, err = sendFileToOrchestrationAPI(processingReq, localMediaPath, logger)
		if err != nil {
			fmt.Printf("❌ File API call failed: %v\n", err)
		}
	} else {
		// Send text message through text processing endpoint
		if contentType == "link" {
			fmt.Printf("📤 Sending link message to API: %s\n", content)
		} else {
			fmt.Printf("📤 Sending text to API: %s\n", content)
		}
		response, err = sendTextToOrchestrationAPI(processingReq, logger)
		if err != nil {
			if contentType == "link" {
				fmt.Printf("❌ Link API call failed: %v\n", err)
			} else {
				fmt.Printf("❌ Text API call failed: %v\n", err)
			}
		}
	}

	if err != nil {
		logger.Errorf("Failed to process message through API: %v", err)
		return
	}

	if response.Success {
		chatType := "Private"
		if isGroup {
			if messageContext.IsChannel {
				chatType = "Channel"
			} else {
				chatType = "Group"
			}
		}

		if mediaType != "" {
			baseMessage := fmt.Sprintf("[PROCESSED %s] %s in %s: [%s: %s] %s (ID: %s)",
				chatType, senderName, chatName, mediaType, filename, content, response.StoredID)
			fmt.Printf("✅ %s\n", formatContextMessage(messageContext, baseMessage))

			if localMediaPath != "" {
				fmt.Printf("💾 Media saved to: %s (%d bytes)\n", localMediaPath, mediaSize)
			}
			if response.FileStoragePath != "" {
				fmt.Printf("🗄️  Stored in API: %s\n", response.FileStoragePath)
			}
		} else {
			// Show different message for link vs text
			var baseMessage string
			if contentType == "link" {
				baseMessage = fmt.Sprintf("[PROCESSED %s LINK] %s in %s: %s (ID: %s)",
					chatType, senderName, chatName, content, response.StoredID)
			} else {
				baseMessage = fmt.Sprintf("[PROCESSED %s] %s in %s: %s (ID: %s)",
					chatType, senderName, chatName, content, response.StoredID)
			}
			fmt.Printf("✅ %s\n", formatContextMessage(messageContext, baseMessage))
		}

		if response.ProcessingJobID != "" {
			fmt.Printf("🔄 Processing Job ID: %s\n", response.ProcessingJobID)
		}

		// Additional context logging
		if messageContext.IsGroup {
			fmt.Printf("👥 Group context: %s\n", messageContext.GroupName)
		} else if messageContext.IsChannel {
			fmt.Printf("🔔 Channel context: %s\n", messageContext.ChannelName)
		}
	} else {
		logger.Errorf("Processing failed: %s", response.Error)
	}
}

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

// fileExists checks if a file exists and is not a directory
func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}

func getChatName(client *whatsmeow.Client, jid types.JID, isGroup bool) string {
	if isGroup {
		groupInfo, err := client.GetGroupInfo(jid)
		if err == nil && groupInfo.Name != "" {
			return groupInfo.Name
		}
		return fmt.Sprintf("Group %s", jid.User)
	} else {
		contact, err := client.Store.Contacts.GetContact(context.Background(), jid)
		if err == nil && contact.FullName != "" {
			return contact.FullName
		}
		return jid.User
	}
}

func getSenderName(client *whatsmeow.Client, jid types.JID) string {
	contact, err := client.Store.Contacts.GetContact(context.Background(), jid)
	if err == nil && contact.FullName != "" {
		return contact.FullName
	}
	return jid.User
}

func extractTextContent(msg *waProto.Message) string {
	if msg == nil {
		return ""
	}

	if text := msg.GetConversation(); text != "" {
		return text
	} else if extendedText := msg.GetExtendedTextMessage(); extendedText != nil {
		return extendedText.GetText()
	}

	return ""
}

func extractMediaInfo(msg *waProto.Message) (mediaType string, filename string, url string, mediaKey []byte, fileSHA256 []byte, fileEncSHA256 []byte, fileLength uint64) {
	if msg == nil {
		return "", "", "", nil, nil, nil, 0
	}

	timestamp := time.Now().Format("20060102_150405")

	if img := msg.GetImageMessage(); img != nil {
		return "image", fmt.Sprintf("image_%s.jpg", timestamp),
			img.GetURL(), img.GetMediaKey(), img.GetFileSHA256(), img.GetFileEncSHA256(), img.GetFileLength()
	}

	if vid := msg.GetVideoMessage(); vid != nil {
		return "video", fmt.Sprintf("video_%s.mp4", timestamp),
			vid.GetURL(), vid.GetMediaKey(), vid.GetFileSHA256(), vid.GetFileEncSHA256(), vid.GetFileLength()
	}

	if aud := msg.GetAudioMessage(); aud != nil {
		return "audio", fmt.Sprintf("audio_%s.ogg", timestamp),
			aud.GetURL(), aud.GetMediaKey(), aud.GetFileSHA256(), aud.GetFileEncSHA256(), aud.GetFileLength()
	}

	if doc := msg.GetDocumentMessage(); doc != nil {
		filename := doc.GetFileName()
		if filename == "" {
			filename = fmt.Sprintf("document_%s", timestamp)
		}
		return "document", filename,
			doc.GetURL(), doc.GetMediaKey(), doc.GetFileSHA256(), doc.GetFileEncSHA256(), doc.GetFileLength()
	}

	return "", "", "", nil, nil, nil, 0
}

func main() {
	logger := waLog.Stdout("Client", "INFO", true)
	fmt.Println("🚀 Starting WhatsApp Bridge with Enhanced Group/Channel Detection...")

	// Ensure file storage directory exists
	if err := ensureFileStorageDirectory(); err != nil {
		logger.Errorf("Failed to create file storage directory: %v", err)
		return
	}
	// Check API health before starting
	if err := checkAPIHealth(); err != nil {
		logger.Errorf("API health check failed: %v", err)
		fmt.Println("❌ Please ensure the Processing API is running:")
		fmt.Printf("   - Processing API: %s\n", PROCESSING_API_URL)
		fmt.Println("   - Start with: python processing_storage_api.py")
		return
	}

	dbLog := waLog.Stdout("Database", "INFO", true)

	if err := os.MkdirAll("store", 0755); err != nil {
		logger.Errorf("Failed to create store directory: %v", err)
		return
	}

	fmt.Println("✅ Storage directories initialized")
	fmt.Printf("📁 File storage base: %s\n", FILE_STORAGE_BASE)
	fmt.Printf("🔗 Orchestration API: %s\n", ORCHESTRATION_API_URL)
	fmt.Printf("🔧 Processing API: %s\n", PROCESSING_API_URL)
	fmt.Printf("💾 Storage API: %s\n", STORAGE_API_URL)

	container, err := sqlstore.New(context.Background(), "sqlite3", "file:store/whatsapp.db?_foreign_keys=on", dbLog)
	if err != nil {
		logger.Errorf("Failed to connect to database: %v", err)
		return
	}

	deviceStore := container.NewDevice()
	fmt.Println("🔄 Created new device - QR code login required")

	client := whatsmeow.NewClient(deviceStore, logger)
	if client == nil {
		logger.Errorf("Failed to create WhatsApp client")
		return
	}

	client.AddEventHandler(func(evt interface{}) {
		switch v := evt.(type) {
		case *events.Message:
			handleIncomingMessage(client, v, logger)
		case *events.Connected:
			fmt.Println("✅ Connected to WhatsApp successfully!")
			fmt.Println("📱 Listening for INCOMING messages with enhanced group/channel detection...")
		case *events.LoggedOut:
			fmt.Println("⚠️  Device logged out - will need to scan QR code again")
		case *events.Disconnected:
			fmt.Println("📴 Disconnected from WhatsApp")
		}
	})

	fmt.Println("📱 Preparing QR code for WhatsApp login...")

	qrChan, _ := client.GetQRChannel(context.Background())
	err = client.Connect()
	if err != nil {
		logger.Errorf("Failed to connect: %v", err)
		return
	}

	fmt.Println("\n🔳 Please scan this QR code with your WhatsApp app:")
	fmt.Println("   1. Open WhatsApp on your phone")
	fmt.Println("   2. Go to Settings > Linked Devices")
	fmt.Println("   3. Tap 'Link a Device'")
	fmt.Println("   4. Scan the QR code below:")

	connected := false
	for evt := range qrChan {
		if evt.Event == "code" {
			qrterminal.GenerateHalfBlock(evt.Code, qrterminal.L, os.Stdout)
			fmt.Println("\n⏳ Waiting for QR code scan...")
		} else if evt.Event == "success" {
			fmt.Println("✅ QR code scanned successfully!")
			connected = true
			break
		} else if evt.Event == "timeout" {
			fmt.Println("⏰ QR code expired, generating new one...")
		}
	}

	if !connected {
		logger.Errorf("Failed to connect via QR code")
		return
	}

	fmt.Println("🔄 Stabilizing connection...")
	time.Sleep(3 * time.Second)

	if !client.IsConnected() {
		logger.Errorf("Failed to establish stable connection")
		return
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("📨 Processing incoming messages through API pipeline")
	fmt.Printf("💾 Local media storage: %s/[phone_number]/[filename]\n", FILE_STORAGE_BASE)
	fmt.Println("🗄️  API storage: Managed by Storage API")
	fmt.Println("🔄 Pipeline: WhatsApp → Orchestration → Processing → Storage")
	fmt.Println("🌐 API Documentation:")
	fmt.Printf("   - Main: %s/docs\n", ORCHESTRATION_API_URL)
	fmt.Printf("   - Processing: %s/docs\n", PROCESSING_API_URL)
	fmt.Printf("   - Storage: %s/docs\n", STORAGE_API_URL)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("Press Ctrl+C to stop the bridge")

	exitChan := make(chan os.Signal, 1)
	signal.Notify(exitChan, syscall.SIGINT, syscall.SIGTERM)

	<-exitChan

	fmt.Println("\n🛑 Shutting down WhatsApp Bridge...")
	client.Disconnect()
	fmt.Println("✅ Disconnected successfully")
}
