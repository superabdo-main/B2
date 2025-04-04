# B2 Load Testing Tool

A powerful, feature-rich load testing tool for API performance evaluation with advanced metrics, customizable load patterns, and comprehensive reporting.

## Features

### Core Capabilities
- **High Concurrency**: Multithreaded with async execution for efficient resource usage
- **Configurable Load**: Adjust thread count, request count, and load patterns
- **Detailed Metrics**: Response times, percentiles, RPS, error rates, and more
- **Rich Reporting**: HTML reports, charts, and console output
- **Live Progress**: Real-time progress tracking with time estimates

### Load Patterns
- **Constant Load**: Steady, continuous request rate
- **Ramp-Up**: Gradually increases load over time
- **Step Load**: Incremental load increases in distinct steps
- **Wave Pattern**: Sinusoidal traffic pattern simulating cyclical load

### Performance Metrics
- **Response Time**: Average, median, and percentiles (p50, p90, p95, p99)
- **Throughput**: Requests per second (RPS)
- **Error Analysis**: Error rates and categorization by status code
- **Response Distribution**: Histogram visualization of response times

### Advanced Features
- **Authentication**: Support for Basic Auth, Bearer Token
- **Custom Headers**: Add any HTTP headers to requests
- **Request Timeouts**: Configurable timeout handling
- **Think Time**: Simulate realistic user behavior with pauses between requests
- **Error Handling**: Detailed error tracking and categorization

## Installation

### Requirements
- Python 3.7+
- Dependencies listed in requirements.txt

### Setup
1. Clone the repository:
```bash
git clone https://github.com/superabdo-main/B2
cd B2
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from b2 import B2, Options

# Create test options
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=100,
    num_threads=4
)

# Initialize and configure client
client = B2()
client.set_options(options)

# Execute test
results = await client.execute_request()

# View results
print(f"Success Rate: {results['success_rate']}%")
print(f"Average Latency: {results['avg_latency']}s")
```

### Command Line Interface
```bash
python example_cli.py --url https://api.example.com/endpoint --method GET --num-requests 100 --num-threads 4
```

## Test Types

### 1. Load Test
Tests system behavior under expected load conditions.

```python
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=100,
    num_threads=4
)
```

### 2. Stress Test
Push the system to its limits to find breaking points.

```python
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=1000,
    num_threads=20
)
```

### 3. Endurance Test
Test system stability over longer durations.

```python
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=5000,
    num_threads=8,
    save_results=True
)
```

### 4. Spike Test
Simulate sudden traffic surges.

```python
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=500,
    num_threads=30
)
```

### 5. Load Pattern Tests
Test with different traffic patterns.

```python
# Ramp-up pattern
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=300,
    num_threads=10,
    load_pattern="ramp-up",
    ramp_up_time=15.0
)

# Step pattern
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET", 
    num_requests=300,
    num_threads=15,
    load_pattern="step",
    ramp_up_steps=5
)

# Wave pattern
options = Options(
    url="https://api.example.com/endpoint",
    request_method="GET",
    num_requests=400, 
    num_threads=16,
    load_pattern="wave",
    test_duration=60.0
)
```

## Advanced Configuration

### Options Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| url | string | required | API endpoint URL |
| request_method | string | "GET" | HTTP method (GET, POST, PUT, DELETE, PATCH) |
| request_payload | dict | {} | Request data for POST/PUT/PATCH |
| output_name | string | "output.json" | Output filename |
| output_path | string | "./output" | Directory for output files |
| num_requests | int | 1 | Number of requests to send |
| save_results | bool | False | Whether to save results to disk |
| num_threads | int | 4 | Number of threads for concurrent requests |
| load_pattern | string | "constant" | Pattern type: constant, ramp-up, step, wave |
| ramp_up_time | float | None | Time (seconds) to ramp up load |
| ramp_up_steps | int | 5 | Number of steps for step pattern |
| test_duration | float | None | Override test duration (seconds) |
| headers | dict | {} | Custom HTTP headers |
| timeout | float | 30.0 | Request timeout in seconds |
| think_time | float | None | Time to wait between requests (seconds) |
| think_time_distribution | string | "constant" | Distribution for think time |
| auth_type | string | None | Authentication type: basic, bearer |
| auth_credentials | dict | {} | Auth credentials (username/password or token) |
| detailed_reporting | bool | True | Enable detailed reports |

### Authentication Example

```python
# Basic Auth
options = Options(
    url="https://api.example.com/secure",
    auth_type="basic",
    auth_credentials={
        "username": "user",
        "password": "pass"
    }
)

# Bearer Token
options = Options(
    url="https://api.example.com/secure",
    auth_type="bearer",
    auth_credentials={
        "token": "your-token-here"
    }
)
```

## Example Files

The repository includes several example files:

1. **example_cli.py**: Command-line interface
2. **example_tests.py**: Test type examples (load, stress, endurance, spike)
3. **advanced_examples.py**: Advanced features (metrics, load patterns, auth)

Run examples with:
```bash
python example_tests.py
python advanced_examples.py
```

## Output and Reports

B2 generates several types of output:

1. **Console Output**: Live progress and result summaries
2. **JSON Results**: Raw test data for further analysis
3. **HTML Reports**: Comprehensive test results with metrics
4. **Histogram Charts**: Visual distribution of response times

Reports are saved to the configured output directory.

## Project Structure

```
B2/
├── b2/
│   ├── __init__.py       # Package initialization
│   ├── b2.py             # Main B2 client implementation
│   └── options.py        # Options configuration class
├── example_cli.py        # Command-line interface
├── example_tests.py      # Test type examples
├── advanced_examples.py  # Advanced features examples
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
