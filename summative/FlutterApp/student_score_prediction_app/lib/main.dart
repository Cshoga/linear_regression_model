import 'package:flutter/material.dart';
import 'prediction_page.dart';

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
