import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

// --- CONFIGURATION: PASTE YOUR SUPABASE CREDENTIALS HERE ---
const String _supabaseUrl = 'https://tzojpimhvhbgsrfurgkd.supabase.co';
const String _supabaseAnonKey = 'sb_publishable_P9WmOozqLj3DopVGVZzNjw_gcOE7jUn';
const String _modelPath = 'assets/models/porthole_w8a32.tflite';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Supabase connection before launching the UI
  await Supabase.initialize(
    url: _supabaseUrl,
    anonKey: _supabaseAnonKey,
  );

  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: PotholeMapperApp(),
  ));
}

class PotholeMapperApp extends StatefulWidget {
  const PotholeMapperApp({super.key});

  @override
  State<PotholeMapperApp> createState() => _PotholeMapperAppState();
}

class _PotholeMapperAppState extends State<PotholeMapperApp> {
  final YOLOViewController _yoloController = YOLOViewController();
  final _supabase = Supabase.instance.client;

  int _tabIndex = 0; // 0 = Snap & Predict, 1 = Pothole Map

  // Camera & Prediction State
  Uint8List? _cleanPhotoBytes;
  Uint8List? _segmentedPhotoBytes;
  bool _isProcessing = false;
  String? _statusMessage;

  // Latest Frame Detections & Area Metrics
  int _latestCount = 0;
  double _latestCoverageRatio = 0.0;
  double _estimatedAreaCm2 = 0.0;
  final double _cameraHeightCm = 100.0; // Default 1m holding height

  // Locked GPS Coordinates for the snapped photo
  Position? _snappedPosition;

  @override
  void initState() {
    super.initState();
    _ensureLocationPermission();
  }

  /// Requests runtime GPS permission on startup
  Future<void> _ensureLocationPermission() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      await Geolocator.requestPermission();
    }
  }

  /// Computes fraction of frame (0.0 to 1.0) covered by a single detection's mask
  double _computeMaskCoverage(YOLOResult result) {
    final mask = result.mask;
    if (mask != null && mask.isNotEmpty && mask.first.isNotEmpty) {
      int activePixels = 0;
      int totalPixels = 0;
      for (final row in mask) {
        for (final val in row) {
          totalPixels++;
          if (val > 0.5) activePixels++;
        }
      }
      return totalPixels > 0 ? activePixels / totalPixels : 0.0;
    }
    return (result.boundingBox.width * result.boundingBox.height).toDouble();
  }

  /// Converts frame coverage ratio to physical cm² via pinhole geometry
  double _estimateRealWorldCm2(double coverageRatio, double cameraHeightCm) {
    const hHalfRad = (65.0 / 2.0) * (math.pi / 180.0);
    const vHalfRad = (50.0 / 2.0) * (math.pi / 180.0);
    final groundWidthCm = 2.0 * cameraHeightCm * math.tan(hHalfRad);
    final groundHeightCm = 2.0 * cameraHeightCm * math.tan(vHalfRad);
    return coverageRatio * (groundWidthCm * groundHeightCm);
  }

  String _classifySeverity(double areaCm2) {
    if (areaCm2 >= 1500) return 'severe';
    if (areaCm2 >= 400) return 'moderate';
    return 'minor';
  }

  /// STEP 1: Snap clean photo in-app + lock current GPS coordinates
  Future<void> _snapPicture() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Acquiring GPS & capturing frame...';
    });

    try {
      // Capture clean photo from the viewfinder
      final bytes = await _yoloController.capturePhoto(withOverlays: false);

      // Lock current GPS coordinates
      Position? pos;
      final perm = await Geolocator.checkPermission();
      if (perm == LocationPermission.whileInUse || perm == LocationPermission.always) {
        pos = await Geolocator.getCurrentPosition(
          locationSettings: const LocationSettings(accuracy: LocationAccuracy.high),
        );
      }

      if (bytes != null && mounted) {
        setState(() {
          _cleanPhotoBytes = bytes;
          _segmentedPhotoBytes = null;
          _snappedPosition = pos;
          _isProcessing = false;
          _statusMessage = pos != null
              ? 'GPS locked: ${pos.latitude.toStringAsFixed(5)}, ${pos.longitude.toStringAsFixed(5)}'
              : 'Warning: GPS unavailable (enable location services)';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _statusMessage = 'Capture error: $e';
        });
      }
    }
  }

  /// STEP 2: Send to model for segmentation, compute area, and upload to Supabase
  Future<void> _sendToModelAndUpload() async {
    setState(() {
      _isProcessing = true;
      _statusMessage = 'Running segmentation & uploading to database...';
    });

    try {
      // Grab composited segmented image from the shared YOLOView session
      final segmented = await _yoloController.capturePhoto(withOverlays: true);
      final areaCm2 = _estimateRealWorldCm2(_latestCoverageRatio, _cameraHeightCm);
      final coveragePct = _latestCoverageRatio * 100.0;
      final severity = _classifySeverity(areaCm2);

      String uploadStatus = 'Segmented locally (no GPS to upload).';

      // Insert row into Supabase if GPS was acquired and at least 1 pothole detected
      if (_snappedPosition != null && _latestCount > 0) {
        await _supabase.from('pothole_detections').insert({
          'latitude': _snappedPosition!.latitude,
          'longitude': _snappedPosition!.longitude,
          'pothole_count': _latestCount,
          'estimated_area_cm2': double.parse(areaCm2.toStringAsFixed(1)),
          'coverage_percent': double.parse(coveragePct.toStringAsFixed(2)),
          'severity': severity,
        });
        uploadStatus = 'Uploaded to Supabase! ($severity • ${areaCm2.toStringAsFixed(0)} cm²)';
      } else if (_latestCount == 0) {
        uploadStatus = 'No potholes detected in frame — nothing uploaded.';
      }

      if (mounted) {
        setState(() {
          _segmentedPhotoBytes = segmented;
          _estimatedAreaCm2 = areaCm2;
          _isProcessing = false;
          _statusMessage = uploadStatus;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isProcessing = false;
          _statusMessage = 'Upload failed: $e';
        });
      }
    }
  }

  void _resetToViewfinder() {
    setState(() {
      _cleanPhotoBytes = null;
      _segmentedPhotoBytes = null;
      _snappedPosition = null;
      _statusMessage = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(_tabIndex == 0 ? 'Pothole Inspector' : 'Mapped Potholes'),
      ),
      body: IndexedStack(
        index: _tabIndex,
        children: [
          // =========================================================
          // TAB 0: SNAP -> SEND TO MODEL -> UPLOAD TO SUPABASE
          // =========================================================
          Stack(
            children: [
              // Clean viewfinder (showOverlays: false keeps live masks hidden until Send)
              YOLOView(
                modelPath: _modelPath,
                task: YOLOTask.segment,
                useGpu: false,
                //showOverlays: false,
                controller: _yoloController,
                confidenceThreshold: 0.25,
                onResult: (results) {
                  double ratio = 0.0;
                  for (final r in results) {
                    ratio += _computeMaskCoverage(r);
                  }
                  _latestCount = results.length;
                  _latestCoverageRatio = ratio;
                },
              ),

              // Viewfinder Shutter Button
              if (_cleanPhotoBytes == null && _segmentedPhotoBytes == null)
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 32.0),
                    child: FloatingActionButton.large(
                      onPressed: _isProcessing ? null : _snapPicture,
                      child: _isProcessing
                          ? const CircularProgressIndicator()
                          : const Icon(Icons.camera_alt),
                    ),
                  ),
                ),

              // Review Screen: Clean Photo + Retake / Send to Model
              if (_cleanPhotoBytes != null && _segmentedPhotoBytes == null)
                Container(
                  color: Colors.black,
                  child: Column(
                    children: [
                      Expanded(
                        child: Center(
                          child: Image.memory(_cleanPhotoBytes!, fit: BoxFit.contain),
                        ),
                      ),
                      if (_statusMessage != null)
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
                          child: Text(
                            _statusMessage!,
                            style: const TextStyle(color: Colors.white70, fontSize: 13),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Row(
                          children: [
                            Expanded(
                              child: OutlinedButton.icon(
                                onPressed: _isProcessing ? null : _resetToViewfinder,
                                icon: const Icon(Icons.close),
                                label: const Text('Retake'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: ElevatedButton.icon(
                                onPressed: _isProcessing ? null : _sendToModelAndUpload,
                                icon: const Icon(Icons.cloud_upload),
                                label: const Text('Send to Model'),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

              // Result Screen: Segmented Output + DB Confirmation
              if (_segmentedPhotoBytes != null)
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
                          _statusMessage ??
                              'Detected $_latestCount pothole(s) • ${_estimatedAreaCm2.toStringAsFixed(0)} cm²',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                        child: SizedBox(
                          width: double.infinity,
                          child: ElevatedButton.icon(
                            onPressed: _resetToViewfinder,
                            icon: const Icon(Icons.refresh),
                            label: const Text('Snap Another Pothole'),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
            ],
          ),

          // =========================================================
          // TAB 1: INTERACTIVE OPENSTREETMAP OF LOGGED POTHOLES
          // =========================================================
          PotholeMapTab(isActive: _tabIndex == 1),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _tabIndex,
        onTap: (index) => setState(() => _tabIndex = index),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.camera_alt), label: 'Snap & Log'),
          BottomNavigationBarItem(icon: Icon(Icons.map), label: 'Pothole Map'),
        ],
      ),
    );
  }
}

// ============================================================================
// MAP WIDGET: FETCHES ROWS FROM SUPABASE & PLOTS SEVERITY PINS
// ============================================================================
class PotholeMapTab extends StatefulWidget {
  final bool isActive;
  const PotholeMapTab({super.key, required this.isActive});

  @override
  State<PotholeMapTab> createState() => _PotholeMapTabState();
}

class _PotholeMapTabState extends State<PotholeMapTab> {
  final _supabase = Supabase.instance.client;
  List<Map<String,dynamic>> _records = [];
  bool _loading = false;

  // Default map center: Abuja, Nigeria (auto-centers on latest pothole once loaded)
  LatLng _center = const LatLng(9.0765, 7.3986);

  @override
  void initState() {
    super.initState();
    _fetchDetections();
  }

  @override
  void didUpdateWidget(covariant PotholeMapTab oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Automatically refresh pins whenever the user switches to the Map tab
    if (widget.isActive && !oldWidget.isActive) {
      _fetchDetections();
    }
  }

  Future<void> _fetchDetections() async {
    setState(() => _loading = true);
    try {
      final response = await _supabase
          .from('pothole_detections')
          .select()
          .order('created_at', ascending: false);

      final rows = List<Map<String, dynamic>>.from(response);
      if (mounted) {
        setState(() {
          _records = rows;
          if (rows.isNotEmpty) {
            _center = LatLng(
              (rows.first['latitude'] as num).toDouble(),
              (rows.first['longitude'] as num).toDouble(),
            );
          }
          _loading = false;
        });
      }
    } catch (e) {
      debugPrint('Error fetching Supabase detections: $e');
      if (mounted) setState(() => _loading = false);
    }
  }

  Color _pinColor(String? severity) {
    switch (severity) {
      case 'severe':
        return Colors.red;
      case 'moderate':
        return Colors.orange;
      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        FlutterMap(
          options: MapOptions(
            initialCenter: _center,
            initialZoom: 14.0,
          ),
          children: [
            TileLayer(
              urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              userAgentPackageName: 'com.example.porthole',
            ),
            MarkerLayer(
              markers: _records.map((row) {
                final lat = (row['latitude'] as num).toDouble();
                final lon = (row['longitude'] as num).toDouble();
                final area = (row['estimated_area_cm2'] as num?)?.toDouble() ?? 0.0;
                final severity = row['severity'] as String?;

                return Marker(
                  point: LatLng(lat, lon),
                  width: 40,
                  height: 40,
                  child: GestureDetector(
                    onTap: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                            'Severity: ${severity ?? "unknown"} | '
                            'Area: ${area.toStringAsFixed(0)} cm² | '
                            '(${lat.toStringAsFixed(5)}, ${lon.toStringAsFixed(5)})',
                          ),
                          duration: const Duration(seconds: 3),
                        ),
                      );
                    },
                    child: Icon(
                      Icons.location_pin,
                      color: _pinColor(severity),
                      size: 38,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
        ),

        // Top Summary Banner + Manual Refresh Button
        Positioned(
          top: 12,
          left: 12,
          right: 12,
          child: Card(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Total Mapped Potholes: ${_records.length}',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  IconButton(
                    icon: _loading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.refresh),
                    onPressed: _loading ? null : _fetchDetections,
                  ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}