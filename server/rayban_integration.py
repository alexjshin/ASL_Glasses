import credentials
import mediapipe as mp
from tensorflow.keras.models import load_model
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class InstagramLiveStreamMonitor:
    def __init__(self, model, acc_user, acc_pass, stream_url=None):
        """
        Initialize the Instagram Live Stream Monitor for real-time ASL Translation.

        Args:
            model (_type_): Trained LSTM Neural Network model
            acc_user (_type_): Instagram account username for livestream
            acc_pass (_type_): Instagram account password for livestream
        """
        # Initialize Instagram account credentials
        self.username = acc_user
        self.password = acc_pass

        # Load Model
        self.model = model
        
        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize Video Source to None
        self.video_source = None
        self.browser = None

    def connect_to_ig_live(self):
        """
        Connect to the Instagram Live stream using the provided credentials.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        try:
            print("Attempting to connect to Instagram Live...")
            
            # Set up Chrome options for Selenium
            chrome_options = Options()
            # Run in headless mode if you don't want to see the browser window
            # chrome_options.add_argument("--headless")
            chrome_options.add_argument("--use-fake-ui-for-media-stream")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize Chrome browser
            self.browser = webdriver.Chrome(options=chrome_options)
            
            # Log in to Instagram
            print("Logging in to Instagram...")
            self.browser.get("https://www.instagram.com/accounts/login/")
            
            # Wait for login page to load and accept cookies if needed
            try:
                cookie_button = WebDriverWait(self.browser, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'Allow')]"))
                )
                cookie_button.click()
            except:
                print("No cookie consent needed or already accepted.")
            
            # Enter username and password
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            
            username_field = self.browser.find_element(By.NAME, "username")
            password_field = self.browser.find_element(By.NAME, "password")
            
            username_field.send_keys(self.username)
            password_field.send_keys(self.password)
            
            # Submit login form
            login_button = self.browser.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for login to complete
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'Search') or contains(text(), 'Explore')]"))
            )
            
            print("Successfully logged in to Instagram")
        except Exception as e:
            print(f"Error connecting to Instagram Live: {e}")
        
        print("Failed to connect to Instagram Live")
        return False
            

def main():
    # Load Model
    model = load_model('Models/02_lstm_model.h5')
    model.load_weights('Models/02_model.weights.h5')
    
    acc_user = credentials.ACC_USER
    acc_pass = credentials.ACC_PW

    monitor = InstagramLiveStreamMonitor(
        model,
        acc_user,
        acc_pass
    )
    monitor.connect_to_ig_live()

if __name__ == "__main__":
    main()