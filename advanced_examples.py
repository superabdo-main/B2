import asyncio
import os
import json
from datetime import datetime
from b2 import B2, Options


async def percentile_metrics_example(url):
    """
    Advanced performance metrics example.
    
    Demonstrates capturing and analyzing detailed percentile metrics.
    """
    print("\n===== ADVANCED METRICS EXAMPLE =====")
    print("Running test with detailed performance metrics\n")
    
    options = Options(
        url=url,
        request_method="GET",
        num_requests=200,
        num_threads=8,
        save_results=True,
        output_name="advanced_metrics_result.json",
        output_path="./advanced_results",
        detailed_reporting=True  # Enable detailed metrics
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    # Print percentile metrics
    print("\nDetailed Performance Metrics:")
    print(f"Average Latency: {results['avg_latency']:.6f}s")
    print(f"Median (P50): {results['percentiles']['p50']:.6f}s")
    print(f"P90 Latency: {results['percentiles']['p90']:.6f}s")
    print(f"P95 Latency: {results['percentiles']['p95']:.6f}s")
    print(f"P99 Latency: {results['percentiles']['p99']:.6f}s")
    print(f"Requests Per Second (RPS): {results['rps']:.2f}")
    
    return results


async def ramp_up_load_example(url):
    """
    Ramp-up load test example.
    
    Demonstrates gradually increasing load over time.
    """
    print("\n===== RAMP-UP LOAD PATTERN EXAMPLE =====")
    print("Gradually increasing load over time\n")
    
    options = Options(
        url=url,
        request_method="GET",
        num_requests=300,
        num_threads=10,
        load_pattern="ramp-up",
        ramp_up_time=15.0,  # Ramp up over 15 seconds
        save_results=True,
        output_name="ramp_up_test_result.json",
        output_path="./load_patterns"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    return results


async def step_load_example(url):
    """
    Step load test example.
    
    Demonstrates increasing load in distinct steps.
    """
    print("\n===== STEP LOAD PATTERN EXAMPLE =====")
    print("Increasing load in distinct steps\n")
    
    options = Options(
        url=url,
        request_method="GET",
        num_requests=300,
        num_threads=15,
        load_pattern="step",
        ramp_up_steps=5,  # Use 5 distinct steps
        save_results=True,
        output_name="step_load_test_result.json",
        output_path="./load_patterns"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    return results


async def wave_load_example(url):
    """
    Wave load test example.
    
    Demonstrates sinusoidal pattern of traffic.
    """
    print("\n===== WAVE LOAD PATTERN EXAMPLE =====")
    print("Generating wave pattern of traffic\n")
    
    options = Options(
        url=url,
        request_method="GET",
        num_requests=400,
        num_threads=16,
        load_pattern="wave",
        test_duration=60.0,  # Run for 60 seconds
        save_results=True,
        output_name="wave_load_test_result.json",
        output_path="./load_patterns"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    return results


async def authenticated_request_example(url):
    """
    Authenticated API request example.
    
    Demonstrates testing APIs that require authentication.
    """
    print("\n===== AUTHENTICATED REQUEST EXAMPLE =====")
    print("Testing endpoints with authentication\n")
    
    # Example with Bearer token
    options = Options(
        url=url,
        request_method="GET",
        num_requests=50,
        num_threads=4,
        auth_type="bearer",
        auth_credentials={"token": "your_api_token_here"},
        save_results=True,
        output_name="auth_test_result.json",
        output_path="./auth_tests"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    return results


async def error_analysis_example(url):
    """
    Error analysis example.
    
    Demonstrates categorizing and analyzing error responses.
    """
    print("\n===== ERROR ANALYSIS EXAMPLE =====")
    print("Analyzing API error responses\n")
    
    # Deliberately use a URL that might return errors (404, etc.)
    error_url = f"{url}/nonexistent-endpoint"
    
    options = Options(
        url=error_url,
        request_method="GET",
        num_requests=100,
        num_threads=5,
        save_results=True,
        output_name="error_analysis_result.json",
        output_path="./error_tests"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    # Analyze error categories
    print("\nError Distribution:")
    for status, count in results["error_categories"].items():
        print(f"  Status {status}: {count} requests ({count/results['requests_executed']*100:.2f}%)")
    
    return results


async def think_time_example(url):
    """
    Think time example.
    
    Demonstrates simulating real user behavior with think time between requests.
    """
    print("\n===== THINK TIME SIMULATION EXAMPLE =====")
    print("Simulating real user behavior with pauses between requests\n")
    
    options = Options(
        url=url,
        request_method="GET",
        num_requests=80,
        num_threads=4,
        think_time=0.5,  # Add 0.5 second pause between requests
        think_time_distribution="random",  # Random variation in think time
        save_results=True,
        output_name="think_time_result.json",
        output_path="./behavior_tests"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    
    return results


async def compare_load_patterns(url):
    """
    Compare different load patterns.
    
    Runs the same test with different load patterns and compares results.
    """
    print("\n===== LOAD PATTERN COMPARISON =====")
    print("Comparing performance across different load patterns\n")
    
    patterns = ["constant", "ramp-up", "step", "wave"]
    results = {}
    
    for pattern in patterns:
        print(f"\nRunning test with {pattern} load pattern:")
        
        options = Options(
            url=url,
            request_method="GET",
            num_requests=200,
            num_threads=8,
            load_pattern=pattern,
            save_results=True,
            output_name=f"{pattern}_pattern_result.json",
            output_path="./pattern_comparison"
        )
        
        client = B2()
        client.set_options(options=options)
        test_result = await client.execute_request()
        
        results[pattern] = {
            "success_rate": test_result["success_rate"],
            "avg_latency": test_result["avg_latency"],
            "p95_latency": test_result["percentiles"]["p95"],
            "rps": test_result["rps"],
            "total_time": test_result["total_elapsed_time"]
        }
    
    # Ensure output directory exists
    if not os.path.exists("./pattern_comparison"):
        os.makedirs("./pattern_comparison")
        
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./pattern_comparison/pattern_comparison_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print("\nLoad Pattern Comparison:")
    print(f"{'Pattern':<10} {'Success Rate':<15} {'Avg Latency':<15} {'P95 Latency':<15} {'RPS':<10}")
    print("-" * 65)
    
    for pattern, metrics in results.items():
        print(f"{pattern:<10} {metrics['success_rate']:.2f}% {metrics['avg_latency']:.6f}s {metrics['p95_latency']:.6f}s {metrics['rps']:.2f}")
    
    return results


async def async_main():
    """
    Main function to run the examples.
    
    Uncomment the examples you want to run.
    """
    # Configure your test target
    url = "https://innovators-backend.vercel.app"  # Example API endpoint
    
    # Run individual examples
    await percentile_metrics_example(url)
    # await ramp_up_load_example(url)
    # await step_load_example(url)
    # await wave_load_example(url)
    # await authenticated_request_example(url)
    # await error_analysis_example(url)
    # await think_time_example(url)
    # await compare_load_patterns(url)
    
    # Or run all examples sequentially
    # await run_all_examples(url)


async def run_all_examples(url):
    """Run all examples with pauses between them."""
    print("\n===== RUNNING ALL ADVANCED EXAMPLES =====")
    
    examples = [
        percentile_metrics_example,
        ramp_up_load_example,
        step_load_example,
        wave_load_example,
        authenticated_request_example,
        error_analysis_example,
        think_time_example,
        compare_load_patterns
    ]
    
    for example in examples:
        await example(url)
        print("\nPausing between examples...\n")
        await asyncio.sleep(3)


def main():
    """Entry point for running the examples."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main() 