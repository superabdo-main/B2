from typing import Optional, Dict, List, Tuple, Counter
from .options import Options
import asyncio
import aiohttp
import time
import json
import os
import concurrent.futures
from threading import Thread
import sys
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class B2:
    """
    B2 client for handling API requests.

    Handles configuration and execution of API requests
    based on provided options using multithreading.
    """

    def __init__(self) -> None:
        """Initialize B2 client with default configuration."""
        super().__init__()

        # API configuration
        self._url: Optional[str] = None
        self._request_method: Optional[str] = None
        self._request_payload: Optional[Dict] = None
        self._headers: Optional[Dict] = None
        self._timeout: Optional[float] = None

        # Output configuration
        self._output_name: Optional[str] = None
        self._output_path: Optional[str] = None
        self._save_results: Optional[bool] = False

        # Statistics
        self._num_requests: int = 0
        self._success_rate: float = 0.0
        self._avg_latency: float = 0.0
        self._response_results: Dict = {}
        
        # Advanced metrics
        self._percentiles: Dict[str, float] = {}
        self._rps: float = 0.0
        self._error_rate: float = 0.0
        self._error_categories: Dict[int, int] = {}
        self._latency_distribution: Dict = {}
        
        # Thread configuration
        self._num_threads: int = 4  # Default number of threads
        self._threads_results: List = []

        # Timing information
        self._start_time: float = 0
        self._completed_requests: int = 0
        self._total_elapsed_time: float = 0
        
        # Test configuration
        self._load_pattern: str = "constant"  # constant, ramp-up, wave
        self._ramp_up_time: Optional[float] = None
        self._ramp_up_steps: int = 10
        self._test_duration: Optional[float] = None
        
        # Behavior configuration
        self._think_time: float = 0.0
        self._think_time_distribution: str = "uniform"
        
        # async session
        self._session: Optional[aiohttp.ClientSession] = None

    def set_options(self, options: Options) -> None:
        """
        Configure client with the provided options.

        Args:
            options: Configuration options for requests and output
        """
        # Basic request configuration
        self._url = options.url
        self._request_method = options.request_method
        self._request_payload = options.request_payload
        self._headers = options.headers.copy()
        self._timeout = options.timeout
        
        # If authentication is configured, add auth headers
        if options.has_authentication:
            auth_headers = options.get_auth_headers()
            self._headers.update(auth_headers)
        
        # Output configuration
        self._output_name = options.output_name
        self._output_path = options.output_path
        self._save_results = options.save_results
        
        # Load test configuration
        self._num_requests = options.num_requests
        self._num_threads = options.num_threads
        self._load_pattern = options.load_pattern
        self._ramp_up_time = options.ramp_up_time
        self._ramp_up_steps = options.ramp_up_steps
        self._test_duration = options.test_duration
        
        # Behavior configuration
        self._think_time = options.think_time
        self._think_time_distribution = options.think_time_distribution

    @property
    def request_count(self) -> int:
        """Return the number of requests made."""
        return self._num_requests
        
    @property
    def num_threads(self) -> int:
        """Return the number of threads."""
        return self._num_threads
        
    @num_threads.setter
    def num_threads(self, value: int) -> None:
        """Set the number of threads."""
        if value < 1:
            raise ValueError("Number of threads must be at least 1")
        self._num_threads = value

    def _validate_inputs(self) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If inputs are invalid
        """
        if not self._url or not self._request_method:
            raise ValueError(
                "Client not properly configured. Call set_options first.")

    async def execute_request(self) -> Dict:
        """
        Execute API request based on configured options.

        Returns:
            Dict containing API response data

        Raises:
            ValueError: If client is not properly configured
        """
        self._validate_inputs()
        
        self._start_time = time.perf_counter()
        self._completed_requests = 0
        
        print(f"Starting load test with {self._num_threads} threads and {self._num_requests} requests")
        print(f"Load pattern: {self._load_pattern}")
        
        # Start the timer display in a separate thread
        timer_thread = Thread(target=self._display_timer)
        timer_thread.daemon = True
        timer_thread.start()
        
        # Execute appropriate load pattern
        if self._load_pattern == "constant":
            await self._execute_constant_load()
        elif self._load_pattern == "ramp-up":
            await self._execute_ramp_up_load()
        elif self._load_pattern == "step":
            await self._execute_step_load()
        elif self._load_pattern == "wave":
            await self._execute_wave_load()
        else:
            # Default to constant load
            await self._execute_constant_load()
        
        # Process and aggregate results
        self._total_elapsed_time = time.perf_counter() - self._start_time
        self._process_thread_results()
        
        return {
            "status": "success", 
            "message": "Load test completed",
            "requests_executed": self._num_requests,
            "success_rate": self._success_rate,
            "error_rate": self._error_rate,
            "error_categories": self._error_categories,
            "avg_latency": self._avg_latency,
            "percentiles": self._percentiles,
            "rps": self._rps,
            "total_elapsed_time": self._total_elapsed_time,
            "latency_distribution": self._latency_distribution,
            "load_pattern": self._load_pattern
        }
    
    def _run_thread_requests(self, num_requests: int, thread_id: int) -> Dict:
        """
        Execute requests in a separate thread.
        
        Args:
            num_requests: Number of requests to execute
            thread_id: Thread identifier
            
        Returns:
            Dictionary with thread results
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async load testing in this thread
            return loop.run_until_complete(
                self._thread_async_requests(num_requests, thread_id)
            )
        finally:
            loop.close()
    
    async def _thread_async_requests(self, num_requests: int, thread_id: int) -> Dict:
        """
        Execute async requests within a thread.
        
        Args:
            num_requests: Number of requests to execute
            thread_id: Thread identifier
            
        Returns:
            Dictionary with thread results
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            thread_start_time = time.perf_counter()
            
            for i in range(num_requests):
                tasks.append(self._make_request(session))
            
            results = await asyncio.gather(*tasks)
            thread_duration = time.perf_counter() - thread_start_time
            
            # Collect thread results
            responses = []
            statuses = []
            latencies = []
            
            for res in results:
                response_data, status, latency = res
                if self._save_results and response_data:
                    responses.append(response_data)
                statuses.append(status)
                latencies.append(latency)
                
            
            thread_success_rate = statuses.count(200) / len(statuses) * 100 if statuses else 0
            thread_avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            print(f"\nThread {thread_id}: {num_requests} requests, " 
                  f"Success Rate: {thread_success_rate:.2f}%, "
                  f"Avg Latency: {thread_avg_latency:.3f}s")
            
            return {
                "thread_id": thread_id,
                "requests": num_requests,
                "responses": responses,
                "statuses": statuses,
                "latencies": latencies,
                "success_rate": thread_success_rate,
                "avg_latency": thread_avg_latency,
                "duration": thread_duration
            }

    async def _make_request(self, session: aiohttp.ClientSession) -> tuple:
        """
        Make an API request using the configured options.

        Args:
            session: The aiohttp client session to use

        Returns:
            Tuple containing response data, status code, and latency
        """
        start_time = time.perf_counter()
        try:
            method = self._request_method.upper()
            kwargs = {
                'timeout': aiohttp.ClientTimeout(total=self._timeout)
            }
            
            # Add headers if set
            if hasattr(self, '_headers') and self._headers:
                kwargs['headers'] = self._headers
                
            # Add payload for appropriate methods
            if method in ["POST", "PUT", "PATCH"] and self._request_payload:
                kwargs["json"] = self._request_payload
                
            async with session.request(method, self._url, **kwargs) as response:
                duration = time.perf_counter() - start_time
                # Update the completed requests counter
                self._completed_requests += 1
                
                response_data = None
                if self._save_results:
                    try:
                        response_data = await response.json()
                    except:
                        try:
                            response_data = await response.text()
                        except:
                            response_data = {'error': 'Failed to parse response'}
                
                return response_data, response.status, duration
                
        except aiohttp.ClientConnectorError as e:
            duration = time.perf_counter() - start_time
            print(f"Connection error: {str(e)}")
            return None, 0, duration
        except aiohttp.ServerTimeoutError as e:
            duration = time.perf_counter() - start_time
            print(f"Timeout error: {str(e)}")
            return None, 408, duration
        except Exception as e:
            duration = time.perf_counter() - start_time
            print(f"Request error: {str(e)}")
            return None, 0, duration
    
    def _process_thread_results(self) -> None:
        """Process and aggregate results from all threads."""
        if not self._threads_results:
            return
            
        all_responses = []
        all_statuses = []
        all_latencies = []
        error_counter = Counter()
        
        for thread_result in self._threads_results:
            if self._save_results:
                all_responses.extend(thread_result.get("responses", []))
            all_statuses.extend(thread_result.get("statuses", []))
            all_latencies.extend(thread_result.get("latencies", []))
        
        # Calculate overall statistics
        successful_requests = all_statuses.count(200)
        total_requests = len(all_statuses)
        
        self._success_rate = successful_requests / total_requests * 100 if total_requests else 0
        self._error_rate = 100 - self._success_rate
        
        # Categorize errors by status code
        for status in all_statuses:
            if status != 200:
                error_counter[status] += 1
        
        self._error_categories = dict(error_counter)
        
        # Calculate latency statistics
        if all_latencies:
            self._avg_latency = sum(all_latencies) / len(all_latencies)
            
            # Calculate percentiles
            all_latencies.sort()
            self._percentiles = {
                "p50": self._calculate_percentile(all_latencies, 50),
                "p90": self._calculate_percentile(all_latencies, 90),
                "p95": self._calculate_percentile(all_latencies, 95),
                "p99": self._calculate_percentile(all_latencies, 99)
            }
            
            # Calculate requests per second
            self._rps = total_requests / self._total_elapsed_time if self._total_elapsed_time > 0 else 0
            
            # Generate latency distribution histogram
            self._generate_latency_distribution(all_latencies)
        
        # Store response results if saving is enabled
        if self._save_results:
            self._response_results = all_responses
        
        # Format the total elapsed time
        elapsed_time_str = self._format_time(self._total_elapsed_time)
            
        print(f"\nOverall Results:")
        print(f"Total Requests: {total_requests}")
        print(f"Success Rate: {self._success_rate:.2f}%")
        print(f"Error Rate: {self._error_rate:.2f}%")
        print(f"RPS (Requests Per Second): {self._rps:.2f}")
        print(f"Avg Latency: {self._avg_latency:.6f}s")
        print(f"Latency Percentiles:")
        print(f"  P50: {self._percentiles.get('p50', 0):.6f}s")
        print(f"  P90: {self._percentiles.get('p90', 0):.6f}s")
        print(f"  P95: {self._percentiles.get('p95', 0):.6f}s")
        print(f"  P99: {self._percentiles.get('p99', 0):.6f}s")
        print(f"Total Elapsed Time: {elapsed_time_str}")
        
        if self._error_categories:
            print("\nError Distribution:")
            for status, count in self._error_categories.items():
                print(f"  Status {status}: {count} requests ({count/total_requests*100:.2f}%)")
        
        # Save results if enabled
        if self._save_results:
            self._save_response_sync()
            self._generate_report()
    
    def _calculate_percentile(self, sorted_values: List[float], percentile: int) -> float:
        """
        Calculate percentile value from sorted list.
        
        Args:
            sorted_values: List of values sorted in ascending order
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0
            
        index = (len(sorted_values) - 1) * percentile / 100
        
        # Handle exact match
        if index.is_integer():
            return sorted_values[int(index)]
            
        # Interpolate between values
        lower_index = math.floor(index)
        upper_index = math.ceil(index)
        
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        
        # Linear interpolation
        fraction = index - lower_index
        return lower_value + (upper_value - lower_value) * fraction
    
    def _generate_latency_distribution(self, latencies: List[float]) -> None:
        """
        Generate latency distribution histogram.
        
        Args:
            latencies: List of request latencies in seconds
        """
        if not latencies:
            self._latency_distribution = {}
            return
            
        # Keep latencies in seconds for better consistency
        latencies_s = latencies.copy()
        
        # Calculate histogram bins
        min_latency = min(latencies_s)
        max_latency = max(latencies_s)
        
        # Use appropriate bin size based on range
        range_s = max_latency - min_latency
        
        if range_s <= 0.01:  # Small range - use 0.0005s bins
            bin_size = 0.0005
        elif range_s <= 0.1:  # Medium range - use 0.005s bins
            bin_size = 0.005
        elif range_s <= 1.0:  # Large range - use 0.05s bins
            bin_size = 0.05
        else:  # Very large range - use 0.1s bins
            bin_size = 0.1
            
        # Create histogram
        num_bins = math.ceil(range_s / bin_size) + 1
        hist, bin_edges = np.histogram(latencies_s, bins=num_bins)
        
        # Store distribution in a more readable format
        distribution = {}
        for i, count in enumerate(hist):
            bin_start = round(bin_edges[i], 6)
            bin_end = round(bin_edges[i+1], 6)
            distribution[f"{bin_start}-{bin_end}s"] = int(count)
            
        self._latency_distribution = distribution
        
        # If saving results, generate histogram chart
        if self._save_results:
            self._generate_histogram_chart(latencies_s, bin_edges, hist)
    
    def _generate_histogram_chart(self, latencies_s: List[float], bin_edges: np.ndarray, hist: np.ndarray) -> None:
        """
        Generate and save histogram chart for latency distribution.
        
        Args:
            latencies_s: List of latencies in seconds
            bin_edges: Bin edges for histogram
            hist: Histogram counts
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(latencies_s, bins=bin_edges, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Response Time (s)')
            plt.ylabel('Number of Requests')
            plt.title('Response Time Distribution')
            plt.grid(True, alpha=0.3)
            
            # Add percentile markers
            percentiles = {
                'p50': self._percentiles.get('p50', 0),
                'p90': self._percentiles.get('p90', 0),
                'p99': self._percentiles.get('p99', 0)
            }
            
            colors = {'p50': 'green', 'p90': 'orange', 'p99': 'red'}
            
            for label, value in percentiles.items():
                plt.axvline(x=value, color=colors[label], linestyle='--', 
                            label=f'{label.upper()}: {value:.6f}s')
            
            plt.legend()
            
            # Ensure output directory exists
            os.makedirs(self._output_path, exist_ok=True)
            
            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(self._output_path, f"latency_histogram_{timestamp}.png")
            plt.savefig(chart_path)
            plt.close()
            
            print(f"\nLatency histogram saved to: {chart_path}")
        except Exception as e:
            print(f"Error generating histogram chart: {str(e)}")
    
    def _generate_report(self) -> None:
        """Generate comprehensive HTML report with test results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self._output_path, f"report_{timestamp}.html")
            
            # Create report data
            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "url": self._url,
                "method": self._request_method,
                "total_requests": self._num_requests,
                "concurrency": self._num_threads,
                "success_rate": f"{self._success_rate:.2f}%",
                "error_rate": f"{self._error_rate:.2f}%",
                "rps": f"{self._rps:.2f}",
                "avg_latency": f"{self._avg_latency:.6f}s",
                "p50": f"{self._percentiles.get('p50', 0):.6f}s",
                "p90": f"{self._percentiles.get('p90', 0):.6f}s",
                "p95": f"{self._percentiles.get('p95', 0):.6f}s",
                "p99": f"{self._percentiles.get('p99', 0):.6f}s",
                "total_time": f"{self._total_elapsed_time:.2f}s",
                "errors": self._error_categories,
                "distribution": self._latency_distribution
            }
            
            # Simple HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>B2 Load Test Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #333; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    .metric {{ margin-bottom: 10px; }}
                    .metric-name {{ font-weight: bold; width: 180px; display: inline-block; }}
                    .metric-value {{ color: #444; }}
                    .section {{ margin-top: 30px; background: #f9f9f9; padding: 20px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart-container {{ margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>B2 Load Test Report</h1>
                    <div class="section">
                        <h2>Test Summary</h2>
                        <div class="metric">
                            <span class="metric-name">Timestamp:</span> 
                            <span class="metric-value">{report_data["timestamp"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">URL:</span> 
                            <span class="metric-value">{report_data["url"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Method:</span> 
                            <span class="metric-value">{report_data["method"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Total Requests:</span> 
                            <span class="metric-value">{report_data["total_requests"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Concurrency:</span> 
                            <span class="metric-value">{report_data["concurrency"]} threads</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Total Test Time:</span> 
                            <span class="metric-value">{report_data["total_time"]}</span>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Performance Metrics</h2>
                        <div class="metric">
                            <span class="metric-name">Success Rate:</span> 
                            <span class="metric-value">{report_data["success_rate"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Error Rate:</span> 
                            <span class="metric-value">{report_data["error_rate"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Requests Per Second:</span> 
                            <span class="metric-value">{report_data["rps"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Average Latency:</span> 
                            <span class="metric-value">{report_data["avg_latency"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">Median (P50):</span> 
                            <span class="metric-value">{report_data["p50"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">P90 Latency:</span> 
                            <span class="metric-value">{report_data["p90"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">P95 Latency:</span> 
                            <span class="metric-value">{report_data["p95"]}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-name">P99 Latency:</span> 
                            <span class="metric-value">{report_data["p99"]}</span>
                        </div>
                    </div>
            """
            
            # Add error distribution section if there are errors
            if report_data["errors"]:
                html_content += f"""
                    <div class="section">
                        <h2>Error Distribution</h2>
                        <table>
                            <tr>
                                <th>Status Code</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                """
                
                for status, count in report_data["errors"].items():
                    percentage = count / report_data["total_requests"] * 100
                    html_content += f"""
                            <tr>
                                <td>{status}</td>
                                <td>{count}</td>
                                <td>{percentage:.2f}%</td>
                            </tr>
                    """
                
                html_content += """
                        </table>
                    </div>
                """
            
            # Add latency distribution section
            html_content += f"""
                    <div class="section">
                        <h2>Response Time Distribution</h2>
                        <table>
                            <tr>
                                <th>Latency Range</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """
            
            for latency_range, count in report_data["distribution"].items():
                percentage = count / report_data["total_requests"] * 100
                html_content += f"""
                        <tr>
                            <td>{latency_range}</td>
                            <td>{count}</td>
                            <td>{percentage:.2f}%</td>
                        </tr>
                """
            
            html_content += """
                        </table>
                        
                        <div class="chart-container">
                            <p>See the latency histogram image in the output directory for visualization.</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write HTML report
            with open(report_path, "w") as f:
                f.write(html_content)
                
            print(f"\nDetailed HTML report saved to: {report_path}")
            
        except Exception as e:
            print(f"Error generating HTML report: {str(e)}")
    
    def _save_response_sync(self) -> None:
        """
        Save the response to the output file (synchronous version).
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path, exist_ok=True)
            
        # Construct the full file path by joining directory and filename
        file_path = os.path.join(self._output_path, self._output_name)
        
        try:
            with open(file_path, "w") as f:
                json.dump(self._response_results, f, indent=2)
            print(f"Response saved to {file_path}")
        except Exception as e:
            print(f"Error saving response: {str(e)}")
            
    # Keeping the async methods for backward compatibility
    async def create_session(self) -> None:
        """
        Create the async session (deprecated).
        """
        pass
        
    async def start_loading(self) -> None:
        """
        Start the loading process (deprecated).
        """
        pass
        
    async def close_session(self) -> None:
        """
        Close the async session (deprecated).
        """
        pass
        
    async def save_response(self) -> None:
        """
        Save the response to the output file (deprecated).
        """
        self._save_response_sync()
        
    def _display_timer(self) -> None:
        """Display the elapsed time and estimated remaining time."""
        try:
            while self._completed_requests < self._num_requests:
                elapsed_time = time.perf_counter() - self._start_time
                
                # Calculate the estimated remaining time
                if self._completed_requests > 0:
                    avg_time_per_request = elapsed_time / self._completed_requests
                    remaining_requests = self._num_requests - self._completed_requests
                    estimated_time_remaining = avg_time_per_request * remaining_requests
                else:
                    estimated_time_remaining = 0
                
                # Format the time strings
                elapsed_str = self._format_time(elapsed_time)
                remaining_str = self._format_time(estimated_time_remaining)
                
                # Calculate progress percentage
                progress_percent = (self._completed_requests / self._num_requests) * 100 if self._num_requests > 0 else 0
                
                # Clear the current line and print the status
                sys.stdout.write(f"\rProgress: {progress_percent:.1f}% | Completed: {self._completed_requests}/{self._num_requests} | Elapsed: {elapsed_str} | Remaining: {remaining_str}")
                sys.stdout.flush()
                
                # Sleep for a short time to avoid high CPU usage
                time.sleep(0.5)
                
        except Exception as e:
            print(f"\nError in timer display: {str(e)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
        
    async def _execute_constant_load(self) -> None:
        """Execute constant load test pattern."""
        # Split requests among threads
        requests_per_thread = max(1, self._num_requests // self._num_threads)
        remainder = self._num_requests % self._num_threads
        
        # Launch threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            futures = []
            for i in range(self._num_threads):
                # Distribute remainder requests among first threads
                thread_requests = requests_per_thread + (1 if i < remainder else 0)
                if thread_requests > 0:
                    futures.append(
                        executor.submit(
                            self._run_thread_requests, 
                            thread_requests, 
                            i
                        )
                    )
            
            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                self._threads_results.append(future.result())
                
    async def _execute_ramp_up_load(self) -> None:
        """Execute ramp-up load test pattern."""
        # Default ramp-up time if not specified
        ramp_up_time = self._ramp_up_time if self._ramp_up_time is not None else 30.0
        
        print(f"Ramping up load over {ramp_up_time} seconds")
        
        # Calculate threads to add in each step
        num_steps = 10  # Divide ramp-up into 10 steps
        threads_per_step = max(1, self._num_threads // num_steps)
        
        # Calculate requests per thread
        requests_per_thread = max(1, self._num_requests // self._num_threads)
        
        # Accumulate results
        time_per_step = ramp_up_time / num_steps
        active_threads = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            futures = []
            
            for step in range(num_steps):
                # Add threads for this step
                threads_to_add = min(threads_per_step, self._num_threads - active_threads)
                
                for i in range(threads_to_add):
                    thread_id = active_threads + i
                    futures.append(
                        executor.submit(
                            self._run_thread_requests, 
                            requests_per_thread, 
                            thread_id
                        )
                    )
                
                active_threads += threads_to_add
                print(f"Ramp-up: {active_threads}/{self._num_threads} threads active")
                
                # Wait for the step duration unless we're at the last step
                if step < num_steps - 1:
                    await asyncio.sleep(time_per_step)
            
            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                self._threads_results.append(future.result())
                
    async def _execute_step_load(self) -> None:
        """Execute step load test pattern with distinct steps."""
        steps = self._ramp_up_steps
        
        print(f"Executing step load pattern with {steps} steps")
        
        # Calculate how many threads to use in each step
        threads_per_step = [int(self._num_threads * (i + 1) / steps) for i in range(steps)]
        requests_per_thread = max(1, self._num_requests // self._num_threads)
        
        for step, num_threads in enumerate(threads_per_step):
            print(f"\nStep {step + 1}/{steps}: Running with {num_threads} threads")
            
            step_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                
                for i in range(num_threads):
                    futures.append(
                        executor.submit(
                            self._run_thread_requests, 
                            requests_per_thread // steps,  # Distribute requests among steps
                            i + (step * self._num_threads) # Unique thread ID
                        )
                    )
                
                # Wait for step to complete
                for future in concurrent.futures.as_completed(futures):
                    step_results.append(future.result())
            
            # Process step results
            self._threads_results.extend(step_results)
            
            # Add pause between steps if not the last step
            if step < steps - 1:
                await asyncio.sleep(2)  # 2 second pause between steps
                
    async def _execute_wave_load(self) -> None:
        """Execute wave load test pattern (sinusoidal load)."""
        wave_cycles = 3  # Number of wave cycles
        total_duration = self._test_duration or 60.0  # Default 1 minute if not specified
        
        print(f"Executing wave load pattern with {wave_cycles} cycles over {total_duration} seconds")
        
        # Calculate time per cycle
        time_per_cycle = total_duration / wave_cycles
        
        # Calculate steps per cycle 
        steps_per_cycle = 8  # 8 points to define the wave
        time_per_step = time_per_cycle / steps_per_cycle
        
        # Calculate minimum and maximum number of threads
        min_threads = max(1, int(self._num_threads * 0.2))  # 20% of max
        
        requests_per_thread = max(1, self._num_requests // (self._num_threads * steps_per_cycle * wave_cycles))
        
        for cycle in range(wave_cycles):
            for step in range(steps_per_cycle):
                # Calculate number of threads for this step (sinusoidal pattern)
                # sin(x) ranges from -1 to 1, so we adjust to range from min_threads to max_threads
                phase = (step / steps_per_cycle) * 2 * math.pi  # 0 to 2π
                sin_value = math.sin(phase)  # -1 to 1
                normalized = (sin_value + 1) / 2  # 0 to 1
                
                active_threads = min_threads + int((self._num_threads - min_threads) * normalized)
                
                print(f"\nWave cycle {cycle+1}/{wave_cycles}, step {step+1}/{steps_per_cycle}: " 
                      f"Running with {active_threads} threads")
                
                step_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=active_threads) as executor:
                    futures = []
                    
                    for i in range(active_threads):
                        thread_id = i + (cycle * steps_per_cycle * self._num_threads) + (step * self._num_threads)
                        futures.append(
                            executor.submit(
                                self._run_thread_requests, 
                                requests_per_thread,
                                thread_id
                            )
                        )
                    
                    # Wait for the step to complete
                    for future in concurrent.futures.as_completed(futures):
                        step_results.append(future.result())
                
                # Add all results
                self._threads_results.extend(step_results)
                
                # Wait for the step duration unless we're at the last step of the last cycle
                if not (cycle == wave_cycles - 1 and step == steps_per_cycle - 1):
                    await asyncio.sleep(time_per_step)
        
        
        
        
        
