import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

const String _modelPath = 'assets/models/porthole_w8a32.tflite';

void main() => runApp(const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: PotholeApp(),
    ));

class PotholeApp extends StatefulWidget {
  const PotholeApp({super.key});

  @override
  State<PotholeApp> createState() => _PotholeAppState();
}

class _PotholeAppState extends State<PotholeApp> {
  final YOLOViewController _controller = YOLOViewController();

  int _tabIndex = 0; // 0 = Live Stream, 1 = Snap & Segment
  Uint8List? _cleanPhotoBytes;     // Snapped photo awaiting "Send"
  Uint8List? _segmentedPhotoBytes; // Returned segmented image
  int _lastDetectionCount = 0;

  Future<void> _switchTab(int index) async {
    setState(() {
      _tabIndex = index;
      _cleanPhotoBytes = null;
      _segmentedPhotoBytes = null;
    });
    // Toggle live masks ON for Live Stream (0), OFF for clean viewfinder (1)
    await _controller.setShowOverlays(index == 0);
  }

  Future<void> _snapCleanPhoto() async {
    // Grab clean frame without overlays for review
    final bytes = await _controller.capturePhoto(withOverlays: false);
    if (bytes != null && mounted) {
      setState(() {
        _cleanPhotoBytes = bytes;
        _segmentedPhotoBytes = null;
      });
    }
  }

  Future<void> _sendToModel() async {
    // Grab frame with native segmentation masks composited from the same model
    final segmented = await _controller.capturePhoto(withOverlays: true);
    if (segmented != null && mounted) {
      setState(() {
        _segmentedPhotoBytes = segmented;
      });
    }
  }

  void _retake() {
    setState(() {
      _cleanPhotoBytes = null;
      _segmentedPhotoBytes = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_tabIndex == 0 ? 'Live Segmentation' : 'Snap & Segment'),
      ),
      body: Stack(
        children: [
          // SINGLE YOLOView & SINGLE MODEL INSTANCE FOR BOTH TABS
          YOLOView(
            modelPath: _modelPath,
            task: YOLOTask.segment,
            useGpu: false,
            controller: _controller,
            confidenceThreshold: 0.25,
            onResult: (results) {
              _lastDetectionCount = results.length;
            },
          ),

          // --- SNAP TAB OVERLAYS (Tab 1) ---
          if (_tabIndex == 1 && _cleanPhotoBytes == null && _segmentedPhotoBytes == null)
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 28.0),
                child: FloatingActionButton.large(
                  onPressed: _snapCleanPhoto,
                  child: const Icon(Icons.camera_alt),
                ),
              ),
            ),

          // Review Screen: Clean Snapped Image + Retake / Send Buttons
          if (_tabIndex == 1 && _cleanPhotoBytes != null && _segmentedPhotoBytes == null)
            Container(
              color: Colors.black,
              child: Column(
                children: [
                  Expanded(
                    child: Center(
                      child: Image.memory(_cleanPhotoBytes!, fit: BoxFit.contain),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _retake,
                            icon: const Icon(Icons.close),
                            label: const Text('Retake'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: _sendToModel,
                            icon: const Icon(Icons.send),
                            label: const Text('Send to Model'),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

          // Result Screen: Segmented Output Image
          if (_tabIndex == 1 && _segmentedPhotoBytes != null)
            Container(
              color: Colors.black,
              child: Column(
                children: [
                  Expanded(
                    child: Center(
                      child: Image.memory(_segmentedPhotoBytes!, fit: BoxFit.contain),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Text(
                      'Detected $_lastDetectionCount pothole(s)',
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: _retake,
                        icon: const Icon(Icons.refresh),
                        label: const Text('Take Another Picture'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _tabIndex,
        onTap: _switchTab,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.videocam), label: 'Live Stream'),
          BottomNavigationBarItem(icon: Icon(Icons.camera), label: 'Snap Photo'),
        ],
      ),
    );
  }
}