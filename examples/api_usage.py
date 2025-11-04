"""
Example usage of the AI Parts Recommendation API.

This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
from typing import Dict, Any


class RecommendationAPIClient:
    """Client for interacting with the AI Parts Recommendation API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/api/health/")
        return response.json()
    
    def generate_recommendations(
        self,
        vehicle_id: str,
        current_odometer: float,
        customer_complaints: str = None,
        dealer_code: str = "DLR_MUM_01"
    ) -> Dict[str, Any]:
        """
        Generate parts recommendations for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            current_odometer: Current odometer reading in km
            customer_complaints: Optional customer complaints
            dealer_code: Dealer code
            
        Returns:
            API response with recommendations
        """
        payload = {
            "vehicle_id": vehicle_id,
            "current_odometer": current_odometer,
            "customer_complaints": customer_complaints,
            "dealer_code": dealer_code
        }
        
        response = self.session.post(
            f"{self.base_url}/api/recommendations/generate",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def submit_feedback(
        self,
        recommendation_id: int,
        feedback_type: str,
        feedback_reason: str = None,
        alternative_part: str = None,
        actual_cost: float = None
    ) -> Dict[str, Any]:
        """
        Submit feedback on a recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            feedback_type: Type of feedback (ACCEPTED, REJECTED, MODIFIED)
            feedback_reason: Reason for feedback
            alternative_part: Alternative part if applicable
            actual_cost: Actual cost if known
            
        Returns:
            API response
        """
        payload = {
            "recommendation_id": recommendation_id,
            "feedback_type": feedback_type,
            "feedback_reason": feedback_reason,
            "alternative_part": alternative_part,
            "actual_cost": actual_cost
        }
        
        response = self.session.post(
            f"{self.base_url}/api/feedback/submit",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def get_recommendation_history(self, vehicle_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get historical recommendations for a vehicle.
        
        Args:
            vehicle_id: Vehicle identifier
            limit: Maximum number of recommendations
            
        Returns:
            List of historical recommendations
        """
        response = self.session.get(
            f"{self.base_url}/api/recommendations/history/{vehicle_id}",
            params={"limit": limit}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get ML model status."""
        response = self.session.get(f"{self.base_url}/api/health/model")
        return response.json()


def main():
    """Example usage of the API client."""
    print("AI Parts Recommendation API - Example Usage")
    print("=" * 50)
    
    # Initialize client
    client = RecommendationAPIClient()
    
    try:
        # Check API health
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Environment: {health['environment']}")
        print()
        
        # Check model status
        print("2. Checking model status...")
        model_status = client.get_model_status()
        print(f"   Model loaded: {model_status['model_loaded']}")
        print(f"   Model version: {model_status['model_version']}")
        print(f"   Confidence threshold: {model_status['confidence_threshold']}")
        print()
        
        # Generate recommendations
        print("3. Generating recommendations...")
        recommendations = client.generate_recommendations(
            vehicle_id="MH12AB1234",
            current_odometer=15250.5,
            customer_complaints="Brake making noise when stopping",
            dealer_code="DLR_MUM_01"
        )
        
        print(f"   Status: {recommendations['status']}")
        print(f"   Vehicle: {recommendations['vehicle_info']['vehicle_model']}")
        print(f"   Recommendations count: {len(recommendations['recommendations'])}")
        print(f"   Total estimated cost: ₹{recommendations['total_estimated_cost']:.2f}")
        print()
        
        # Display top recommendations
        print("   Top recommendations:")
        for i, rec in enumerate(recommendations['recommendations'][:3], 1):
            print(f"   {i}. {rec['part_name']} - {rec['confidence_score']:.1f}% confidence")
            print(f"      Category: {rec['category']}, Cost: ₹{rec['estimated_cost']:.2f}")
        print()
        
        # Submit feedback (if we have a recommendation ID)
        if recommendations['recommendations']:
            print("4. Submitting feedback...")
            first_rec = recommendations['recommendations'][0]
            feedback = client.submit_feedback(
                recommendation_id=first_rec.get('id', 1),  # Assuming ID exists
                feedback_type="ACCEPTED",
                feedback_reason="Customer approved this recommendation",
                actual_cost=first_rec['estimated_cost'] * 0.95  # 5% discount
            )
            print(f"   Feedback status: {feedback['status']}")
            print(f"   Message: {feedback['message']}")
            print()
        
        # Get recommendation history
        print("5. Getting recommendation history...")
        history = client.get_recommendation_history("MH12AB1234", limit=5)
        print(f"   Found {len(history)} historical recommendations")
        print()
        
        print("✅ API example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure the API server is running on http://localhost:8000")
        print("Start the server with: python run_api.py")


if __name__ == "__main__":
    main()
