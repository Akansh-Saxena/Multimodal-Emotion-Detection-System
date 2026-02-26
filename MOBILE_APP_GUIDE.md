# NeuroSense - Mobile Application Integration Guide

This guide details how to build a final-year cross-platform mobile application (Android/iOS) that acts as a client to the FastAPI AI Backend.

## 1. Client-Server Architecture (API-First Design)
Your mobile application **will not run the heavy ML models locally**. 
Instead, it follows an API-First Client-Server Architecture:
- **Client (Mobile App)**: Handles UI, captures hardware data (Camera/Mic), compresses media, and displays the "Cognitive Telemetry".
- **Server (FastAPI Backend)**: Receives HTTP/WebSocket requests, processes tensors via PyTorch running on a GPU, and returns JSON payload predictions.

---

## 2. Setting Up the Mobile App (Flutter Recommended)

**Flutter (Dart)** is the industry standard for rapidly building beautiful UI for Android and iOS simultaneously. 

### Recommended Flutter Packages (`pubspec.yaml`):
```yaml
dependencies:
  camera: ^0.10.5+5        # Hardware Camera Access
  image_picker: ^1.0.4     # File picking for gallery images
  web_socket_channel: ^2.4.0 # For real-time frame streaming
  http: ^1.1.0             # Standard REST API handling
  image: ^4.1.3            # To compress images before sending
```

---

## 3. Real-Time WebSocket Streaming vs REST (Handling Edge Cases)

Since 4K images will crash your API, your mobile app **must downsample** frames. Furthermore, standard HTTP POST is too slow for "live" emotion detection. You must use **WebSockets** for a persistent, low-latency connection.

### Example Flutter Architecture (Dart Code Stub):

#### Connecting to the WebSocket API
```dart
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:camera/camera.dart';
import 'dart:convert';
import 'package:image/image.dart' as img;

class EmotionStreamService {
  final channel = WebSocketChannel.connect(
    Uri.parse('ws://YOUR_SERVER_IP:8000/ws/predict/vision'),
  );

  void startCameraStream(CameraController cameraController) {
    int frameCount = 0;
    
    // Listen for incoming Emotion Predictions from the Server
    channel.stream.listen((message) {
      final data = jsonDecode(message);
      print("Predicted Emotion: \${data['predicted_emotion']}");
      print("Confidence: \${data['confidence_scores']}");
      print("Latency: \${data['latency_ms']}ms");
    });

    // Capture camera frames
    cameraController.startImageStream((CameraImage image) {
      frameCount++;
      // Important: Only send 1-2 frames per second to prevent DDOSing your own server
      if (frameCount % 15 == 0) {
        // Pseudo-code compression:
        // 1. Convert CameraImage format to standard JPG bytes
        // 2. Downsample resolution (e.g. 300x300)
        List<int> compressedJpgBytes = _compressAndConvertImage(image);
        
        // Stream bytes to FastAPI WebSocket Backend
        channel.sink.add(compressedJpgBytes);
      }
    });
  }

  void dispose() {
    channel.sink.close();
  }
}
```

### Key Mobile Requirements to Validate in Viva:
1. **Camera & Mic Permissions**: The app must request user permissions cleanly. If rejected, show a helpful UI message gracefully.
2. **Media Compression**: Mobile cameras are high resolution. Always compress frames to standard definition JPEG before transmitting over WebSockets to save bandwidth.
3. **Authentication**: Your app should store the JWT Token (from `/token`) in Secure Storage (e.g., `flutter_secure_storage`) and append it to standard HTTP requests.

---

## 4. App-Specific Polish (Standing out in Vivas)

To make your mobile app feel like a true native application rather than just a web wrapper, implement these hardware-specific features:

### A. Haptic Feedback (Vibrations)
Use the `haptic_feedback` (Flutter) or `react-native-haptic-feedback` library.
- **Trigger**: Fire a lightweight `HapticFeedback.lightImpact()` whenever a face is first successfully detected in the frame.
- **Trigger**: Fire a `HapticFeedback.heavyImpact()` when the backend returns a high-confidence "Warning" (e.g., Extreme Anger or Fear) to physically alert the user.

### B. AR Camera Overlays (Bounding Boxes)
Instead of just showing the raw camera feed, draw over it:
1. When your `CameraImage` stream detects a face (using Google ML Kit for on-device fast detection), get the X/Y coordinates of the face.
2. Draw a sleek, rounded rectangle over those coordinates using a `CustomPainter` (Flutter).
3. **Dynamic Coloring**: Map the color of the bounding box to the real-time emotion returned by your WebSocket:
   - Yellow = Happy
   - Blue = Sad
   - Red = Angry
   - Purple = Fear

### C. Swipeable History Logs
Use a package like `flutter_card_swiper` (Tinder-style swipes). When reviewing the SQLite Temporal Emotion logs via the HTTP backend, allow the user to swipe left/right to navigate through their psychological history fluidly.
