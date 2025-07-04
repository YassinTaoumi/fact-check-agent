"""
Command-line interface for the fact-checker system.
"""

import asyncio
import argparse
import json
import sys
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from fact_checker import FactChecker
from config import config

console = Console()

class FactCheckerCLI:
    """Command-line interface for the fact-checker."""
    
    def __init__(self):
        self.fact_checker = FactChecker()
    
    async def check_text(self, text: str, output_format: str = "rich") -> None:
        """Check facts in text and display results."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fact-checking text...", total=None)
            
            try:
                results = await self.fact_checker.check_facts(text)
                progress.update(task, completed=True)
                
                if output_format == "json":
                    self._output_json(results)
                elif output_format == "plain":
                    self._output_plain(results)
                else:
                    self._output_rich(results)
                    
            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Error: {e}[/red]")
    
    def _output_rich(self, results: dict) -> None:
        """Output results in rich format."""
        
        # Header
        success_color = "green" if results['success'] else "red"
        console.print(Panel(
            f"[{success_color}]Fact-Check Results[/{success_color}]",
            subtitle=f"Processed {len(results['claims'])} claims"
        ))
        
        # Overall statistics
        stats = results['overall_statistics']
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"• Total Claims: {stats['total_claims']}")
        console.print(f"• Average Confidence: {stats['avg_confidence']:.2f}")
        console.print(f"• High Confidence Claims: {stats['high_confidence_claims']}")
        console.print(f"• Processing Time: {stats['processing_summary']['total_time']:.2f}s")
        
        # Verdicts summary
        if stats['verdicts_summary']:
            console.print(f"\n[bold]Verdict Distribution:[/bold]")
            for verdict, count in stats['verdicts_summary'].items():
                color = self._get_verdict_color(verdict)
                console.print(f"• {verdict}: [{color}]{count}[/{color}]")
        
        # Individual claims
        if results['results']:
            console.print(f"\n[bold]Detailed Results:[/bold]")
            
            for i, result in enumerate(results['results'], 1):
                self._display_claim_result(result, i)
        
        # Errors
        if results.get('errors'):
            console.print("\n[red][bold]Errors:[/bold][/red]")
            for error in results['errors']:
                console.print(f"• [red]{error}[/red]")
    
    def _display_claim_result(self, result: dict, claim_number: int) -> None:
        """Display a single claim result."""
        verdict = result['verdict']
        confidence = result['confidence']
        
        # Create claim panel
        verdict_color = self._get_verdict_color(verdict)
        
        claim_content = f"""
[bold]Claim:[/bold] {result['claim']}

[bold]Verdict:[/bold] [{verdict_color}]{verdict}[/{verdict_color}]
[bold]Confidence:[/bold] {confidence:.2f}
[bold]Quality:[/bold] {result['quality_assessment'].get('quality_level', 'Unknown')}

[bold]Justification:[/bold]
{result['justification']}
"""
        
        if result['limitations']:
            claim_content += f"\n[bold]Limitations:[/bold]\n{result['limitations']}"
        
        console.print(Panel(
            claim_content.strip(),
            title=f"Claim {claim_number}",
            border_style=verdict_color
        ))
        
        # Source details table
        if result['source_details']:
            table = Table(title="Sources")
            table.add_column("Title", style="cyan")
            table.add_column("URL", style="blue")
            table.add_column("Credibility", justify="center")
            table.add_column("Relevance", justify="center")
            
            for source in result['source_details'][:5]:  # Show top 5 sources
                table.add_row(
                    source['title'][:50] + "..." if len(source['title']) > 50 else source['title'],
                    source['url'][:40] + "..." if len(source['url']) > 40 else source['url'],
                    f"{source['credibility_score']:.2f}",
                    f"{source['relevance_score']:.2f}"
                )
            
            console.print(table)
        
        console.print()  # Add spacing
    
    def _get_verdict_color(self, verdict: str) -> str:
        """Get color for verdict display."""
        colors = {
            'TRUE': 'green',
            'FALSE': 'red',
            'PARTLY_TRUE': 'yellow',
            'UNVERIFIED': 'blue'
        }
        return colors.get(verdict, 'white')
    
    def _output_json(self, results: dict) -> None:
        """Output results in JSON format."""
        print(json.dumps(results, indent=2))
    
    def _output_plain(self, results: dict) -> None:
        """Output results in plain text format."""
        print("FACT-CHECK RESULTS")
        print("=" * 50)
        
        stats = results['overall_statistics']
        print(f"Total Claims: {stats['total_claims']}")
        print(f"Average Confidence: {stats['avg_confidence']:.2f}")
        print(f"Processing Time: {stats['processing_summary']['total_time']:.2f}s")
        print()
        
        for i, result in enumerate(results['results'], 1):
            print(f"CLAIM {i}: {result['claim']}")
            print(f"VERDICT: {result['verdict']}")
            print(f"CONFIDENCE: {result['confidence']:.2f}")
            print(f"JUSTIFICATION: {result['justification']}")
            print("-" * 50)
    
    async def check_file(self, file_path: str, output_format: str = "rich") -> None:
        """Check facts in a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            console.print(f"[blue]Reading file: {file_path}[/blue]")
            await self.check_text(text, output_format)
            
        except FileNotFoundError:
            console.print(f"[red]Error: File '{file_path}' not found[/red]")
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")
    
    def show_info(self) -> None:
        """Show system information."""
        info = self.fact_checker.get_workflow_info()
        
        console.print(Panel("[bold]Fact-Checker System Information[/bold]"))
        
        # Configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        
        for key, value in info['configuration'].items():
            config_table.add_row(key, str(value))
        
        console.print(config_table)
        
        # Workflow steps
        console.print("\n[bold]Workflow Steps:[/bold]")
        for i, step in enumerate(info['workflow_steps'], 1):
            console.print(f"{i}. {step}")
        
        # Components
        console.print("\n[bold]Components:[/bold]")
        for component, description in info['components'].items():
            console.print(f"• {component}: {description}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AI-powered fact-checker")
    parser.add_argument('--version', action='version', version='1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Fact-check text')
    check_parser.add_argument('text', help='Text to fact-check')
    check_parser.add_argument('--format', choices=['rich', 'json', 'plain'], 
                             default='rich', help='Output format')
    
    # File command
    file_parser = subparsers.add_parser('file', help='Fact-check text file')
    file_parser.add_argument('path', help='Path to text file')
    file_parser.add_argument('--format', choices=['rich', 'json', 'plain'], 
                            default='rich', help='Output format')
    
    # Info command
    subparsers.add_parser('info', help='Show system information')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = FactCheckerCLI()
    
    try:
        if args.command == 'check':
            asyncio.run(cli.check_text(args.text, args.format))
        elif args.command == 'file':
            asyncio.run(cli.check_file(args.path, args.format))
        elif args.command == 'info':
            cli.show_info()
        elif args.command == 'interactive':
            asyncio.run(interactive_mode(cli))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def interactive_mode(cli: FactCheckerCLI):
    """Interactive mode for the fact-checker."""
    console.print(Panel(
        "[bold]Interactive Fact-Checker[/bold]\n"
        "Enter text to fact-check, or type 'quit' to exit.",
        title="Welcome"
    ))
    
    while True:
        try:
            text = console.input("\n[cyan]Enter text to fact-check: [/cyan]")
            
            if text.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not text.strip():
                console.print("[red]Please enter some text[/red]")
                continue
            
            await cli.check_text(text)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


if __name__ == "__main__":
    main()
