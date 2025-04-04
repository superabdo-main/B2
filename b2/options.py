from typing import Dict, Optional, List, Any, Union


class Options:
    """Configuration options for API requests and output handling."""
    
    def __init__(
        self, 
        url: str, 
        request_method: str = "GET", 
        request_payload: Dict = None, 
        output_name: str = 'output.json', 
        output_path: str = './output',
        num_requests: int = 1,
        save_results: bool = False,
        num_threads: int = 4,
        load_pattern: str = "constant",
        ramp_up_time: Optional[float] = None,
        ramp_up_steps: int = 5,
        test_duration: Optional[float] = None,
        headers: Dict[str, str] = None,
        timeout: float = 30.0,
        think_time: Optional[float] = None,
        think_time_distribution: str = "constant",
        auth_type: Optional[str] = None,
        auth_credentials: Dict[str, str] = None,
        custom_metrics: List[str] = None,
        detailed_reporting: bool = True
    ) -> None:
        """
        Initialize Options with API request and output configuration.
        
        Args:
            url: API endpoint URL
            request_method: HTTP method (GET, POST, etc.)
            request_payload: Request data as dictionary
            output_name: Filename for output
            output_path: Directory path for output
            num_requests: Number of requests to send
            save_results: Save API response results in JSON file
            num_threads: Number of threads to use for load testing (default: 4)
            load_pattern: Pattern of request generation (constant, ramp-up, step, wave)
            ramp_up_time: Time in seconds to ramp up to full load (for ramp-up pattern)
            ramp_up_steps: Number of steps for step load pattern
            test_duration: Override test duration (in seconds) instead of completing all requests
            headers: Custom HTTP headers to send with each request
            timeout: Request timeout in seconds
            think_time: Time in seconds to wait between requests (simulates user think time)
            think_time_distribution: Distribution pattern for think time (constant, random, normal)
            auth_type: Authentication type (basic, bearer, oauth, etc.)
            auth_credentials: Authentication credentials as dictionary
            custom_metrics: List of custom metrics to track
            detailed_reporting: Whether to generate detailed reports (histograms, etc.)
        """
        super().__init__()
        self._validate_inputs(url, request_method, num_threads, load_pattern)
        
        # Basic request configuration
        self.url = url
        self.request_method = request_method
        self.request_payload = request_payload or {}
        self.headers = headers or {}
        self.timeout = timeout
        
        # Output configuration
        self.output_name = output_name
        self.output_path = output_path
        self.save_results = save_results
        self.detailed_reporting = detailed_reporting
        
        # Load test configuration
        self.num_requests = num_requests
        self.num_threads = num_threads
        self.load_pattern = load_pattern
        self.ramp_up_time = ramp_up_time
        self.ramp_up_steps = ramp_up_steps
        self.test_duration = test_duration
        
        # Behavior configuration
        self.think_time = think_time
        self.think_time_distribution = think_time_distribution
        
        # Authentication
        self.auth_type = auth_type
        self.auth_credentials = auth_credentials or {}
        
        # Metrics configuration
        self.custom_metrics = custom_metrics or []
        
    def _validate_inputs(self, url: str, request_method: str, num_threads: int, load_pattern: str) -> None:
        """
        Validate input parameters.
        
        Args:
            url: API endpoint URL
            request_method: HTTP method
            num_threads: Number of threads
            load_pattern: Load pattern type
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        if request_method.upper() not in valid_methods:
            raise ValueError(f"Request method must be one of {valid_methods}")
            
        if num_threads < 1:
            raise ValueError("Number of threads must be at least 1")
            
        valid_patterns = ["constant", "ramp-up", "step", "wave"]
        if load_pattern not in valid_patterns:
            raise ValueError(f"Load pattern must be one of {valid_patterns}")
    
    @property
    def has_authentication(self) -> bool:
        """Check if authentication is configured."""
        return self.auth_type is not None and bool(self.auth_credentials)
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers based on auth_type.
        
        Returns:
            Dictionary of authentication headers
        """
        if not self.has_authentication:
            return {}
            
        auth_headers = {}
        
        if self.auth_type == "basic":
            import base64
            if "username" in self.auth_credentials and "password" in self.auth_credentials:
                auth_string = f"{self.auth_credentials['username']}:{self.auth_credentials['password']}"
                encoded = base64.b64encode(auth_string.encode()).decode()
                auth_headers["Authorization"] = f"Basic {encoded}"
                
        elif self.auth_type == "bearer":
            if "token" in self.auth_credentials:
                auth_headers["Authorization"] = f"Bearer {self.auth_credentials['token']}"
                
        # Add other auth types as needed
        
        return auth_headers
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert options to dictionary for serialization.
        
        Returns:
            Dictionary representation of options
        """
        return {
            "url": self.url,
            "request_method": self.request_method,
            "num_requests": self.num_requests,
            "num_threads": self.num_threads,
            "load_pattern": self.load_pattern,
            "save_results": self.save_results,
            "detailed_reporting": self.detailed_reporting,
            # Include additional parameters but exclude credentials for security
            "has_auth": self.has_authentication,
            "auth_type": self.auth_type
        }


