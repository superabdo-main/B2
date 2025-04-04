import asyncio
import time
import json
import os
from datetime import datetime
from b2 import B2, Options


async def load_test(url, request_method="GET", payload=None):
    """
    Regular load test - simulates normal expected load.
    
    Tests system behavior under expected load conditions.
    """
    print("\n===== LOAD TEST =====")
    print("Simulating normal expected traffic load\n")
    
    options = Options(
        url=url,
        request_method=request_method,
        request_payload=payload,
        num_requests=100,
        num_threads=4,
        save_results=False
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    return results


async def stress_test(url, request_method="GET", payload=None):
    """
    Stress test - push the system to its limits.
    
    Tests system behavior under heavy load to find breaking points.
    """
    print("\n===== STRESS TEST =====")
    print("Pushing system to its limits with high concurrency\n")
    
    options = Options(
        url=url,
        request_method=request_method,
        request_payload=payload,
        num_requests=1000,
        num_threads=20,  # High thread count for concurrency
        save_results=False
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    return results


async def performance_benchmark(url, request_method="GET", payload=None):
    """
    Performance benchmarking - compare performance metrics.
    
    Runs multiple configurations and saves results for comparison.
    """
    print("\n===== PERFORMANCE BENCHMARK =====")
    print("Running performance benchmark with result comparison\n")
    
    configs = [
        {"name": "Low Concurrency", "threads": 2, "requests": 100},
        {"name": "Medium Concurrency", "threads": 8, "requests": 100},
        {"name": "High Concurrency", "threads": 16, "requests": 100}
    ]
    
    results = {}
    for config in configs:
        print(f"\nRunning benchmark: {config['name']}")
        options = Options(
            url=url,
            request_method=request_method,
            request_payload=payload,
            num_requests=config["requests"],
            num_threads=config["threads"],
            save_results=True,
            output_name=f"benchmark_{config['name'].lower().replace(' ', '_')}.json",
            output_path="./benchmark_results"
        )
        
        client = B2()
        client.set_options(options=options)
        test_result = await client.execute_request()
        results[config["name"]] = {
            "success_rate": test_result["success_rate"],
            "avg_latency": test_result["avg_latency"],
            "total_time": test_result["total_elapsed_time"]
        }
    
    # Create directory if it doesn't exist
    if not os.path.exists("./benchmark_results"):
        os.makedirs("./benchmark_results")
        
    # Save the comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"./benchmark_results/comparison_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nBenchmark Comparison:")
    for name, metrics in results.items():
        print(f"{name}: {metrics['avg_latency']:.3f}s avg latency, {metrics['success_rate']:.1f}% success rate")
    
    return results


async def endurance_test(url, request_method="GET", payload=None):
    """
    Endurance test - tests system over longer duration.
    
    Checks for performance degradation or memory leaks over time.
    """
    print("\n===== ENDURANCE TEST =====")
    print("Testing system stability over longer duration\n")
    
    options = Options(
        url=url,
        request_method=request_method,
        request_payload=payload,
        num_requests=5000,  # Higher request count for longer duration
        num_threads=8,      # Moderate concurrency
        save_results=True,
        output_name="endurance_test_results.json",
        output_path="./test_results"
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    return results


async def spike_test(url, request_method="GET", payload=None):
    """
    Spike test - sudden surge in traffic.
    
    Tests how system handles sudden bursts of high traffic.
    """
    print("\n===== SPIKE TEST =====")
    print("Simulating sudden traffic surge\n")
    
    options = Options(
        url=url,
        request_method=request_method,
        request_payload=payload,
        num_requests=500,
        num_threads=30,  # Very high thread count for traffic spike
        save_results=False
    )
    
    client = B2()
    client.set_options(options=options)
    results = await client.execute_request()
    return results


async def async_main():
    """
    Main function to run different types of tests.
    
    Uncomment the test you want to run or add your own combinations.
    """
    # Configure your test target
    url = "https://innovators-backend.vercel.app"  # Replace with your API endpoint
    method = "GET"                          # HTTP method
    payload = {"param": "value"}            # Request data (for POST/PUT)
    
    # Choose which tests to run
    await load_test(url, method, payload)
    # await stress_test(url, method, payload)
    # await performance_benchmark(url, method, payload)
    # await endurance_test(url, method, payload)
    # await spike_test(url, method, payload)
    
    # Run all tests in sequence
    # await run_all_tests(url, method, payload)


async def run_all_tests(url, method, payload):
    """Run all test types in sequence with pauses between them."""
    print("\n===== RUNNING ALL TEST TYPES =====")
    
    await load_test(url, method, payload)
    print("\nPausing between tests...\n")
    await asyncio.sleep(3)
    
    await stress_test(url, method, payload)
    print("\nPausing between tests...\n")
    await asyncio.sleep(3)
    
    await performance_benchmark(url, method, payload)
    print("\nPausing between tests...\n")
    await asyncio.sleep(3)
    
    await endurance_test(url, method, payload)
    print("\nPausing between tests...\n")
    await asyncio.sleep(3)
    
    await spike_test(url, method, payload)
    

def main():
    """Entry point for the application."""
    asyncio.run(async_main())
    
    
if __name__ == "__main__":
    main()
