#!/usr/bin/env python3
"""
Grok Live Search Utility
Command-line tool for searching with Grok's Live Search capabilities
"""
import asyncio
import argparse
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from providers.xai_provider_advanced import XaiAdvancedProvider

# Load environment variables
load_dotenv()

class GrokSearcher:
    def __init__(self, api_key=None):
        self.provider = XaiAdvancedProvider({'api_key': api_key})
    
    async def search(
        self,
        query: str,
        model: str = "grok-4",
        sources: list = None,
        mode: str = "auto",
        country: str = None,
        from_date: str = None,
        to_date: str = None,
        max_results: int = 20,
        x_handles: list = None,
        exclude_sites: list = None,
        allow_sites: list = None,
        format_output: str = "text"
    ):
        """Perform search with Grok"""
        
        # Build search parameters
        search_params = {
            "mode": mode,
            "return_citations": True,
            "max_search_results": max_results
        }
        
        # Add date filters
        if from_date:
            search_params["from_date"] = from_date
        if to_date:
            search_params["to_date"] = to_date
        
        # Build sources
        if not sources:
            sources = ["web", "x", "news"]
        
        source_configs = []
        for source_type in sources:
            source_config = {"type": source_type}
            
            # Add source-specific parameters
            if source_type in ["web", "news"]:
                if country:
                    source_config["country"] = country
                if exclude_sites:
                    source_config["excluded_websites"] = exclude_sites[:5]
                if allow_sites:
                    source_config["allowed_websites"] = allow_sites[:5]
            
            elif source_type == "x":
                if x_handles:
                    source_config["included_x_handles"] = x_handles[:10]
            
            source_configs.append(source_config)
        
        search_params["sources"] = source_configs
        
        # Perform search
        try:
            result = await self.provider.complete(
                prompt=query,
                model=model,
                search_parameters=search_params,
                temperature=0.7
            )
            
            return result
            
        except Exception as e:
            return f"Error: {e}"
    
    def format_result(self, result: str, format_type: str = "text"):
        """Format search result"""
        if format_type == "json":
            # Extract citations if present
            lines = result.split('\n')
            content_lines = []
            citations = []
            
            in_sources = False
            for line in lines:
                if line.startswith("**Sources:**"):
                    in_sources = True
                    continue
                elif in_sources and line.strip():
                    citations.append(line.strip())
                elif not in_sources:
                    content_lines.append(line)
            
            return json.dumps({
                "content": '\n'.join(content_lines).strip(),
                "citations": citations,
                "timestamp": datetime.now().isoformat()
            }, indent=2)
        
        else:  # text format
            return result

async def main():
    parser = argparse.ArgumentParser(
        description="Grok Live Search - Search the web with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python grok_search.py "latest AI news"
  
  # Search specific sources
  python grok_search.py "OpenAI GPT-4" --sources web news
  
  # Search X (Twitter) posts
  python grok_search.py "AI developments" --sources x --x-handles openai elonmusk
  
  # Search with date range
  python grok_search.py "climate change" --from-date 2024-01-01 --to-date 2024-12-31
  
  # Search specific websites
  python grok_search.py "Python tutorials" --allow-sites python.org docs.python.org
  
  # Search excluding certain sites
  python grok_search.py "best restaurants" --exclude-sites yelp.com reddit.com
  
  # Country-specific search
  python grok_search.py "local news" --country US
  
  # Output as JSON
  python grok_search.py "stock market" --format json
        """
    )
    
    parser.add_argument("query", help="Search query")
    
    parser.add_argument(
        "--model", 
        default="grok-4",
        choices=["grok-4", "grok-3", "grok-3-mini"],
        help="Grok model to use"
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["web", "x", "news"],
        choices=["web", "x", "news", "rss"],
        help="Data sources to search"
    )
    
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "on", "off"],
        help="Search mode"
    )
    
    parser.add_argument(
        "--country",
        help="Country code for web/news search (e.g., US, UK, CA)"
    )
    
    parser.add_argument(
        "--from-date",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--to-date", 
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum search results"
    )
    
    parser.add_argument(
        "--x-handles",
        nargs="+",
        help="X (Twitter) handles to search (max 10)"
    )
    
    parser.add_argument(
        "--exclude-sites",
        nargs="+",
        help="Websites to exclude (max 5)"
    )
    
    parser.add_argument(
        "--allow-sites",
        nargs="+", 
        help="Only search these websites (max 5)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--save",
        help="Save result to file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ Error: XAI_API_KEY environment variable not set", file=sys.stderr)
        print("Please set your xAI API key:", file=sys.stderr)
        print("export XAI_API_KEY='your-api-key-here'", file=sys.stderr)
        sys.exit(1)
    
    # Initialize searcher
    searcher = GrokSearcher(api_key)
    
    if args.verbose:
        print(f"🔍 Searching with Grok {args.model}...")
        print(f"Query: {args.query}")
        print(f"Sources: {', '.join(args.sources)}")
        if args.country:
            print(f"Country: {args.country}")
        if args.from_date or args.to_date:
            print(f"Date range: {args.from_date or 'start'} to {args.to_date or 'end'}")
        print()
    
    try:
        # Perform search
        result = await searcher.search(
            query=args.query,
            model=args.model,
            sources=args.sources,
            mode=args.mode,
            country=args.country,
            from_date=args.from_date,
            to_date=args.to_date,
            max_results=args.max_results,
            x_handles=args.x_handles,
            exclude_sites=args.exclude_sites,
            allow_sites=args.allow_sites
        )
        
        # Format result
        formatted_result = searcher.format_result(result, args.format)
        
        # Output result
        print(formatted_result)
        
        # Save if requested
        if args.save:
            with open(args.save, 'w') as f:
                f.write(formatted_result)
            if args.verbose:
                print(f"\n✓ Result saved to {args.save}", file=sys.stderr)
    
    except KeyboardInterrupt:
        print("\n❌ Search cancelled", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        await searcher.provider.cleanup()

if __name__ == "__main__":
    asyncio.run(main())