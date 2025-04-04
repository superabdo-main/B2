import os
import click
from typing import Dict
import asyncio

from b2 import Options
from b2 import B2


async def async_main(url: str, method: str, num_requests: int, num_threads: int, 
                    payload: str, output_name: str, output_path: str, save_results: bool) -> None:
    """
    Execute API requests using the B2 client.

    Configures and runs the B2 client with the provided command-line options.
    """
    try:
        # Create output directory if saving results
        if save_results and not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            click.echo(f"Created output directory: {output_path}")
            
        # Parse payload from string to dict (simplified version)
        payload_dict: Dict = {}
        if payload and payload != "{}":
            import json
            try:
                payload_dict = json.loads(payload)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON payload format")

        click.echo(f"Starting load test with {num_threads} thread(s) and {num_requests} request(s) to {url}")
        
        # Configure options
        options = Options(
            url=url,
            request_method=method,
            request_payload=payload_dict,
            output_name=output_name,
            output_path=output_path,
            num_requests=num_requests,
            save_results=save_results,
            num_threads=num_threads
        )

        # Initialize and configure client
        client = B2()
        client.set_options(options)

        # Execute request
        click.echo("Load test in progress...")
        response = await client.execute_request()

        # Display results
        click.echo(f"\nLoad test completed successfully")
        click.echo(f"Success Rate: {response.get('success_rate', 0):.2f}%")
        click.echo(f"Avg Latency: {response.get('avg_latency', 0):.3f}s")
        
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        return


@click.command()
@click.option("--url", help="API URL", required=True, type=str)
@click.option("--method", help="HTTP method (GET, POST, PUT, DELETE, PATCH)",
              default="GET", type=str, show_default=True)
@click.option("--num-requests", help="Number of requests to make", default=10, type=int, show_default=True)
@click.option("--num-threads", help="Number of threads to use", default=4, type=int, show_default=True)
@click.option("--save-results", help="Save the API response results in JSON file", is_flag=True, default=False, show_default=True)
@click.option("--payload", help="JSON payload for request", default="{}", type=str)
@click.option("--output-name", help="Output filename", default="response.json",
              type=str, show_default=True)
@click.option("--output-path", help="Output directory path", default="./output",
              type=str, show_default=True)
def main(url: str, method: str, num_requests: int, num_threads: int, payload: str, 
         output_name: str, output_path: str, save_results: bool) -> None:
    """B2 Load Testing Tool - A multithreaded HTTP load tester."""
    asyncio.run(async_main(url, method, num_requests, num_threads, payload, output_name, output_path, save_results))


if __name__ == '__main__':
    main()
