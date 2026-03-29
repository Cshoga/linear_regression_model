import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

const String _apiBaseUrl = 'https://linear-regression-model-pl1a.onrender.com';

void main() {
  runApp(const StudentCounselingApp());
}

class StudentCounselingApp extends StatelessWidget {
  const StudentCounselingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Student Score Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        inputDecorationTheme: const InputDecorationTheme(
          border: OutlineInputBorder(),
          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  bool _isLoading = false;
  String _resultText = '';
  String _flagText = '';
  Color _flagColor = Colors.grey;
  bool _hasError = false;

  final _ageCtrl      = TextEditingController();
  final _studyHrsCtrl = TextEditingController();
  final _attendCtrl   = TextEditingController();
  final _mathCtrl     = TextEditingController();
  final _sciCtrl      = TextEditingController();
  final _engCtrl      = TextEditingController();

  String? _gender          = 'male';
  String? _schoolType      = 'public';
  String? _parentEdu       = 'graduate';
  String? _internet        = 'yes';
  String? _travelTime      = '<15 min';
  String? _extraActivities = 'yes';
  String? _studyMethod     = 'mixed';

  @override
  void dispose() {
    _ageCtrl.dispose();
    _studyHrsCtrl.dispose();
    _attendCtrl.dispose();
    _mathCtrl.dispose();
    _sciCtrl.dispose();
    _engCtrl.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    final age      = int.tryParse(_ageCtrl.text.trim());
    final studyHrs = double.tryParse(_studyHrsCtrl.text.trim());
    final attend   = double.tryParse(_attendCtrl.text.trim());
    final math     = double.tryParse(_mathCtrl.text.trim());
    final science  = double.tryParse(_sciCtrl.text.trim());
    final english  = double.tryParse(_engCtrl.text.trim());

    if (age == null || studyHrs == null || attend == null ||
        math == null || science == null || english == null) {
      setState(() {
        _hasError   = true;
        _resultText = '';
        _flagText   = 'Please fill in all fields with valid numbers.';
        _flagColor  = Colors.red;
      });
      return;
    }

    setState(() {
      _isLoading  = true;
      _hasError   = false;
      _resultText = '';
      _flagText   = '';
    });

    try {
      final body = jsonEncode({
        "age":                   age,
        "gender":                _gender,
        "school_type":           _schoolType,
        "parent_education":      _parentEdu,
        "study_hours":           studyHrs,
        "attendance_percentage": attend,
        "internet_access":       _internet,
        "travel_time":           _travelTime,
        "extra_activities":      _extraActivities,
        "study_method":          _studyMethod,
        "math_score":            math,
        "science_score":         science,
        "english_score":         english,
      });

      final response = await http.post(
        Uri.parse('$_apiBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: body,
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data  = jsonDecode(response.body);
        final score = data['predicted_overall_score'];
        final flag  = data['counseling_flag'] as String;

        Color flagColor = Colors.green;
        if (flag.contains('High risk'))    flagColor = Colors.red;
        else if (flag.contains('At risk')) flagColor = Colors.orange;

        setState(() {
          _isLoading  = false;
          _resultText = 'Predicted Score: ${score.toStringAsFixed(2)} / 100';
          _flagText   = flag;
          _flagColor  = flagColor;
          _hasError   = false;
        });
      } else {
        final data = jsonDecode(response.body);
        setState(() {
          _isLoading  = false;
          _hasError   = true;
          _resultText = '';
          _flagText   = 'Error ${response.statusCode}: ${data['detail'] ?? 'Unknown error'}';
          _flagColor  = Colors.red;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading  = false;
        _hasError   = true;
        _resultText = '';
        _flagText   = 'Network error: $e';
        _flagColor  = Colors.red;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF3F6FB),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
        title: const Text(
          '🎓 Student Score Predictor',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 4,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Fill in the student details below, then tap Predict.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.black54, fontSize: 13),
            ),
            const SizedBox(height: 16),

            // ---- Demographics ----
            _sectionCard(
              title: 'Demographics',
              children: [
                _numericField('Age (10–25)', _ageCtrl, 'e.g. 16'),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Gender',
                  value: _gender,
                  items: const ['male', 'female', 'other'],
                  onChanged: (v) => setState(() => _gender = v),
                ),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'School Type',
                  value: _schoolType,
                  items: const ['public', 'private'],
                  onChanged: (v) => setState(() => _schoolType = v),
                ),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Parent Education',
                  value: _parentEdu,
                  items: const [
                    'graduate', 'high school', 'no formal',
                    'phd', 'post graduate'
                  ],
                  onChanged: (v) => setState(() => _parentEdu = v),
                ),
              ],
            ),

            // ---- Study Habits ----
            _sectionCard(
              title: 'Study Habits',
              children: [
                _numericField('Study Hours / Day (0–12)', _studyHrsCtrl, 'e.g. 4.5'),
                const SizedBox(height: 12),
                _numericField('Attendance % (0–100)', _attendCtrl, 'e.g. 87.5'),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Internet Access',
                  value: _internet,
                  items: const ['yes', 'no'],
                  onChanged: (v) => setState(() => _internet = v),
                ),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Travel Time to School',
                  value: _travelTime,
                  items: const ['<15 min', '15-30 min', '30-60 min', '>60 min'],
                  onChanged: (v) => setState(() => _travelTime = v),
                ),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Extra-Curricular Activities',
                  value: _extraActivities,
                  items: const ['yes', 'no'],
                  onChanged: (v) => setState(() => _extraActivities = v),
                ),
                const SizedBox(height: 12),
                _dropdownField(
                  label: 'Study Method',
                  value: _studyMethod,
                  items: const [
                    'notes', 'textbook', 'online videos',
                    'group study', 'mixed'
                  ],
                  onChanged: (v) => setState(() => _studyMethod = v),
                ),
              ],
            ),

            // ---- Subject Scores ----
            _sectionCard(
              title: 'Subject Scores (0–100)',
              children: [
                _numericField('Math Score',    _mathCtrl, 'e.g. 72.0'),
                const SizedBox(height: 12),
                _numericField('Science Score', _sciCtrl,  'e.g. 68.0'),
                const SizedBox(height: 12),
                _numericField('English Score', _engCtrl,  'e.g. 75.0'),
              ],
            ),

            const SizedBox(height: 8),

            // ---- Predict Button ----
            ElevatedButton(
              onPressed: _isLoading ? null : _predict,
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF1565C0),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                textStyle: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 24,
                      width: 24,
                      child: CircularProgressIndicator(
                        color: Colors.white,
                        strokeWidth: 2.5,
                      ),
                    )
                  : const Text('Predict'),
            ),

            const SizedBox(height: 20),

            // ---- Result Display ----
            if (_resultText.isNotEmpty || _flagText.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border.all(color: _flagColor, width: 2),
                  borderRadius: BorderRadius.circular(14),
                  boxShadow: [
                    BoxShadow(
                      color: _flagColor.withOpacity(0.15),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    if (_resultText.isNotEmpty)
                      Text(
                        _resultText,
                        textAlign: TextAlign.center,
                        style: const TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF1565C0),
                        ),
                      ),
                    if (_resultText.isNotEmpty) const SizedBox(height: 10),
                    Text(
                      _flagText,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: _flagColor,
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }

  Widget _sectionCard({required String title, required List<Widget> children}) {
    return Card(
      elevation: 2,
      margin: const EdgeInsets.only(bottom: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 15,
                color: Color(0xFF1565C0),
              ),
            ),
            const Divider(height: 18),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _numericField(String label, TextEditingController ctrl, String hint) {
    return TextField(
      controller: ctrl,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        filled: true,
        fillColor: const Color(0xFFF8FAFF),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  Widget _dropdownField({
    required String label,
    required String? value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      decoration: InputDecoration(
        labelText: label,
        filled: true,
        fillColor: const Color(0xFFF8FAFF),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
      ),
      items: items.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
      onChanged: onChanged,
    );
  }
}
