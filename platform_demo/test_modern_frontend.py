#!/usr/bin/env python
"""
Test script for Modern Water Quality Frontend
Tests all new features and functionality
"""

import requests
import json
import time
import sys
from urllib.parse import urljoin

class ModernFrontendTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.api_url = urljoin(base_url, "api/")
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        print(f"{status} {test_name}")
        if details:
            print(f"    Details: {details}")
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🔍 Testing API Endpoints...")
        
        # Test stations endpoint
        try:
            response = requests.get(f"{self.api_url}stations", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Stations API", data.get('success', False), 
                            f"Found {len(data.get('data', []))} stations")
            else:
                self.log_test("Stations API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Stations API", False, str(e))
        
        # Test statistics endpoint
        try:
            response = requests.get(f"{self.api_url}statistics/summary", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Statistics API", data.get('success', False))
            else:
                self.log_test("Statistics API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Statistics API", False, str(e))
        
        # Test monitoring endpoints
        try:
            response = requests.get(f"{self.api_url}monitoring/realtime", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Realtime Monitoring API", data.get('success', False))
            else:
                self.log_test("Realtime Monitoring API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Realtime Monitoring API", False, str(e))
        
        try:
            response = requests.get(f"{self.api_url}monitoring/alerts", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Alerts API", data.get('success', False))
            else:
                self.log_test("Alerts API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Alerts API", False, str(e))
        
        try:
            response = requests.get(f"{self.api_url}monitoring/data-quality", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Data Quality API", data.get('success', False))
            else:
                self.log_test("Data Quality API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Data Quality API", False, str(e))
        
        # Test prediction endpoints
        try:
            prediction_data = {
                "Water_Quality_Index": 35.0,
                "COD_mg_L": 20.0,
                "pH": 7.0,
                "Total_Nitrogen_mg_L": 3.0,
                "Total_Phosphorus_mg_L": 0.1,
                "Dissolved_Oxygen_mg_L": 8.0,
                "Ammonia_Nitrogen_mg_L": 0.5,
                "Turbidity_NTU": 5.0
            }
            response = requests.post(f"{self.api_url}predict/pollution_level", 
                                 json=prediction_data, timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Pollution Level Prediction API", data.get('success', False))
            else:
                self.log_test("Pollution Level Prediction API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Pollution Level Prediction API", False, str(e))
        
        # Test chart endpoints
        try:
            response = requests.get(f"{self.api_url}charts/pollution-distribution", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Pollution Distribution Chart API", data.get('success', False))
            else:
                self.log_test("Pollution Distribution Chart API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Pollution Distribution Chart API", False, str(e))
        
        try:
            response = requests.get(f"{self.api_url}charts/province-comparison", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Province Comparison Chart API", data.get('success', False))
            else:
                self.log_test("Province Comparison Chart API", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Province Comparison Chart API", False, str(e))
    
    def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        print("\n🌐 Testing Frontend Accessibility...")
        
        # Test main page access
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for modern elements
                modern_elements = [
                    'modern-nav',
                    'nav-brand', 
                    'nav-menu',
                    'hero-section',
                    'quick-actions',
                    'overview-section',
                    'floating-card',
                    'action-card',
                    'modern-main'
                ]
                
                missing_elements = []
                for element in modern_elements:
                    if element not in content:
                        missing_elements.append(element)
                
                if not missing_elements:
                    self.log_test("Modern Frontend Structure", True, "All modern elements found")
                else:
                    self.log_test("Modern Frontend Structure", False, 
                                f"Missing elements: {missing_elements}")
                
                # Check for CSS files
                css_files = ['modern-design.css', 'style.css']
                missing_css = []
                for css_file in css_files:
                    if css_file not in content:
                        missing_css.append(css_file)
                
                if not missing_css:
                    self.log_test("CSS Files Included", True, "All CSS files found")
                else:
                    self.log_test("CSS Files Included", False, f"Missing CSS: {missing_css}")
                
                # Check for JavaScript files
                js_files = ['modern-app.js', 'map.js', 'monitoring.js', 'prediction.js']
                missing_js = []
                for js_file in js_files:
                    if js_file not in content:
                        missing_js.append(js_file)
                
                if not missing_js:
                    self.log_test("JavaScript Files Included", True, "All JS files found")
                else:
                    self.log_test("JavaScript Files Included", False, f"Missing JS: {missing_js}")
                
            else:
                self.log_test("Frontend Access", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Frontend Access", False, str(e))
    
    def test_page_navigation(self):
        """Test page navigation structure"""
        print("\n📄 Testing Page Navigation...")
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for page sections
                pages = [
                    'dashboard-page',
                    'monitoring-page', 
                    'prediction-page',
                    'analysis-page',
                    'quality-page',
                    'spatial-page'
                ]
                
                missing_pages = []
                for page in pages:
                    if f'id="{page}"' not in content:
                        missing_pages.append(page)
                
                if not missing_pages:
                    self.log_test("Page Structure", True, "All page sections found")
                else:
                    self.log_test("Page Structure", False, f"Missing pages: {missing_pages}")
                
                # Check for navigation items
                nav_items = [
                    'data-page="dashboard"',
                    'data-page="monitoring"',
                    'data-page="prediction"',
                    'data-page="analysis"',
                    'data-page="quality"',
                    'data-page="spatial"'
                ]
                
                missing_nav = []
                for nav_item in nav_items:
                    if nav_item not in content:
                        missing_nav.append(nav_item)
                
                if not missing_nav:
                    self.log_test("Navigation Items", True, "All navigation items found")
                else:
                    self.log_test("Navigation Items", False, f"Missing nav: {missing_nav}")
                
            else:
                self.log_test("Page Navigation Test", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Page Navigation Test", False, str(e))
    
    def test_responsive_design(self):
        """Test responsive design elements"""
        print("\n📱 Testing Responsive Design...")
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for responsive meta tag
                if 'viewport' in content:
                    self.log_test("Viewport Meta Tag", True)
                else:
                    self.log_test("Viewport Meta Tag", False)
                
                # Check for responsive CSS elements
                responsive_indicators = [
                    '@media',
                    'grid-template-columns',
                    'flex-direction',
                    'responsive:'
                ]
                
                responsive_score = 0
                for indicator in responsive_indicators:
                    if indicator in content:
                        responsive_score += 1
                
                if responsive_score >= 2:
                    self.log_test("Responsive CSS", True, f"Found {responsive_score}/{len(responsive_indicators)} indicators")
                else:
                    self.log_test("Responsive CSS", False, f"Only {responsive_score}/{len(responsive_indicators)} indicators")
                
            else:
                self.log_test("Responsive Design Test", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Responsive Design Test", False, str(e))
    
    def test_modern_features(self):
        """Test modern design features"""
        print("\n✨ Testing Modern Features...")
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                content = response.text
                
                # Check for modern CSS features
                modern_features = [
                    'backdrop-filter',
                    'grid',
                    'flexbox',
                    'border-radius',
                    'box-shadow',
                    'transition',
                    'transform'
                ]
                
                modern_score = 0
                for feature in modern_features:
                    if feature in content:
                        modern_score += 1
                
                if modern_score >= 4:
                    self.log_test("Modern CSS Features", True, f"Found {modern_score}/{len(modern_features)} features")
                else:
                    self.log_test("Modern CSS Features", False, f"Only {modern_score}/{len(modern_features)} features")
                
                # Check for animations
                animation_indicators = [
                    '@keyframes',
                    'animation',
                    'transition:',
                    'transform:'
                ]
                
                animation_score = 0
                for indicator in animation_indicators:
                    if indicator in content:
                        animation_score += 1
                
                if animation_score >= 2:
                    self.log_test("Animation Features", True, f"Found {animation_score}/{len(animation_indicators)} indicators")
                else:
                    self.log_test("Animation Features", False, f"Only {animation_score}/{len(animation_indicators)} indicators")
                
                # Check for modern UI elements
                ui_elements = [
                    'floating-card',
                    'action-card',
                    'overview-card',
                    'modern-btn',
                    'indicator-card'
                ]
                
                ui_score = 0
                for element in ui_elements:
                    if element in content:
                        ui_score += 1
                
                if ui_score >= 3:
                    self.log_test("Modern UI Elements", True, f"Found {ui_score}/{len(ui_elements)} elements")
                else:
                    self.log_test("Modern UI Elements", False, f"Only {ui_score}/{len(ui_elements)} elements")
                
            else:
                self.log_test("Modern Features Test", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Modern Features Test", False, str(e))
    
    def test_data_loading(self):
        """Test data loading capabilities"""
        print("\n📊 Testing Data Loading...")
        
        # Test stations data
        try:
            response = requests.get(f"{self.api_url}stations", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success') and data.get('data'):
                    stations = data['data']
                    self.log_test("Stations Data Load", True, f"Loaded {len(stations)} stations")
                    
                    # Check station data structure
                    if len(stations) > 0:
                        sample_station = stations[0]
                        required_fields = ['name', 'province', 'coordinates', 'latest_wqi']
                        missing_fields = [field for field in required_fields 
                                        if field not in sample_station]
                        
                        if not missing_fields:
                            self.log_test("Station Data Structure", True)
                        else:
                            self.log_test("Station Data Structure", False, 
                                        f"Missing fields: {missing_fields}")
                else:
                    self.log_test("Stations Data Load", False, "No data returned")
            else:
                self.log_test("Stations Data Load", False, f"Status: {response.status_code}")
        except Exception as e:
            self.log_test("Stations Data Load", False, str(e))
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("📋 TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  • {result['test']}: {result['details']}")
        
        print("\n" + "="*60)
        
        # Save detailed report
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'results': self.test_results
        }
        
        with open('frontend_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print("📄 Detailed report saved to: frontend_test_report.json")
        print("="*60)
        
        return failed_tests == 0
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Modern Frontend Testing...")
        print("="*60)
        
        # Run all test suites
        self.test_api_endpoints()
        self.test_frontend_accessibility()
        self.test_page_navigation()
        self.test_responsive_design()
        self.test_modern_features()
        self.test_data_loading()
        
        # Generate report
        return self.generate_report()

def main():
    """Main function"""
    print("Modern Water Quality Frontend Tester")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            print("Please make sure the server is running on http://localhost:5000")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server")
        print("Please start the server first: python run.py")
        sys.exit(1)
    
    # Run tests
    tester = ModernFrontendTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
