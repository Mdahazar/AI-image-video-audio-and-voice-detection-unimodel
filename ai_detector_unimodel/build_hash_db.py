# build_hash_db.py
from video_analyzer.detector import VideoDetector

def build_video_hash_database():
    """Build the video hash database"""
    print("🏗️ Building video hash database...")
    detector = VideoDetector()
    detector.build_hash_database()
    print("✅ Video hash database built successfully!")

if __name__ == "__main__":
    build_video_hash_database()