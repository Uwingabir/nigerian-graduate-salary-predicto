import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Nigerian Graduate Salary Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const SplashScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class SplashScreen extends StatelessWidget {
  const SplashScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.green.shade50,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.school,
                size: 100,
                color: Colors.green.shade700,
              ),
              const SizedBox(height: 30),
              Text(
                'Nigerian Graduate',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.green.shade800,
                ),
                textAlign: TextAlign.center,
              ),
              Text(
                'Salary Predictor',
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.green.shade800,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              Text(
                'Predict your earning potential based on education, demographics, and career factors',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.green.shade600,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 50),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const PredictionScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade700,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                ),
                child: const Text(
                  'Start Prediction',
                  style: TextStyle(fontSize: 18),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const AboutScreen(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.grey.shade600,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                ),
                child: const Text(
                  'About the App',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('About'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Mission',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.green.shade800,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'To analyze the factors affecting graduate employment in Nigeria and predict employment outcomes based on education, demographics, and socioeconomic background, helping young Africans make informed career and education decisions.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),
            Text(
              'Problem Statement',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.green.shade800,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'Many Nigerian graduates struggle to find well-paying jobs despite their education. Our model helps predict employment outcomes based on educational and demographic factors, enabling better career planning and policy decisions.',
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 20),
            Text(
              'Solution',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: Colors.green.shade800,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'An AI-powered prediction system that estimates graduate salary and employment probability, helping young Africans make informed education and career choices while identifying factors that improve employment outcomes.',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  
  // Controllers for text fields
  final _ageController = TextEditingController();
  final _yearsController = TextEditingController();
  
  // Dropdown values
  String? _gender;
  String? _region;
  String? _urbanRural;
  String? _incomeLevel;
  String? _fieldOfStudy;
  String? _universityType;
  String? _gpaClass;
  String? _hasPostgrad;
  
  // Result variables
  String _result = '';
  bool _isLoading = false;
  
  // Dropdown options
  final List<String> _genderOptions = ['Male', 'Female'];
  final List<String> _regionOptions = ['North', 'South', 'East'];
  final List<String> _urbanRuralOptions = ['Urban', 'Rural'];
  final List<String> _incomeLevelOptions = ['Low', 'Middle', 'High'];
  final List<String> _fieldOfStudyOptions = [
    'Engineering', 'Business', 'Health Sciences', 'Education', 'Arts', 'Science'
  ];
  final List<String> _universityTypeOptions = ['Federal', 'State', 'Private'];
  final List<String> _gpaClassOptions = [
    'First Class', 'Second Class Upper', 'Second Class Lower', 'Third Class'
  ];
  final List<String> _hasPostgradOptions = ['Yes', 'No'];

  Future<void> _makePrediction() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _result = '';
    });

    try {
      // Using local API for testing
      const String apiUrl = 'http://localhost:8000/predict';
      
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'age': int.parse(_ageController.text),
          'gender': _gender,
          'region': _region,
          'urban_or_rural': _urbanRural,
          'household_income_bracket': _incomeLevel,
          'field_of_study': _fieldOfStudy,
          'university_type': _universityType,
          'gpa_or_class_of_degree': _gpaClass,
          'has_postgrad_degree': _hasPostgrad,
          'years_since_graduation': int.parse(_yearsController.text),
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _result = 'Predicted Salary: ${data['formatted_salary']}\n\n'
                  'Confidence: ${data['model_confidence']}';
        });
      } else {
        setState(() {
          _result = 'Error: ${response.statusCode}\n${response.body}';
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Error: Could not connect to the prediction service.\n'
                'Please check your internet connection or try again later.\n\n'
                'Error details: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildDropdown(String label, String? value, List<String> options, 
                       Function(String?) onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: DropdownButtonFormField<String>(
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        value: value,
        items: options.map((String option) {
          return DropdownMenuItem<String>(
            value: option,
            child: Text(option),
          );
        }).toList(),
        onChanged: onChanged,
        validator: (value) {
          if (value == null || value.isEmpty) {
            return 'Please select $label';
          }
          return null;
        },
      ),
    );
  }

  Widget _buildTextField(String label, TextEditingController controller, 
                        String? Function(String?)? validator) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: TextFormField(
        controller: controller,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        keyboardType: TextInputType.number,
        validator: validator,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Salary Prediction'),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter Your Information',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.green.shade800,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              
              _buildTextField(
                'Age (20-50)',
                _ageController,
                (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter your age';
                  }
                  final age = int.tryParse(value);
                  if (age == null || age < 20 || age > 50) {
                    return 'Age must be between 20 and 50';
                  }
                  return null;
                },
              ),
              
              _buildDropdown('Gender', _gender, _genderOptions, (value) {
                setState(() {
                  _gender = value;
                });
              }),
              
              _buildDropdown('Region', _region, _regionOptions, (value) {
                setState(() {
                  _region = value;
                });
              }),
              
              _buildDropdown('Urban or Rural', _urbanRural, _urbanRuralOptions, (value) {
                setState(() {
                  _urbanRural = value;
                });
              }),
              
              _buildDropdown('Household Income Level', _incomeLevel, _incomeLevelOptions, (value) {
                setState(() {
                  _incomeLevel = value;
                });
              }),
              
              _buildDropdown('Field of Study', _fieldOfStudy, _fieldOfStudyOptions, (value) {
                setState(() {
                  _fieldOfStudy = value;
                });
              }),
              
              _buildDropdown('University Type', _universityType, _universityTypeOptions, (value) {
                setState(() {
                  _universityType = value;
                });
              }),
              
              _buildDropdown('GPA/Class of Degree', _gpaClass, _gpaClassOptions, (value) {
                setState(() {
                  _gpaClass = value;
                });
              }),
              
              _buildDropdown('Has Postgraduate Degree', _hasPostgrad, _hasPostgradOptions, (value) {
                setState(() {
                  _hasPostgrad = value;
                });
              }),
              
              _buildTextField(
                'Years Since Graduation (0-20)',
                _yearsController,
                (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter years since graduation';
                  }
                  final years = int.tryParse(value);
                  if (years == null || years < 0 || years > 20) {
                    return 'Years must be between 0 and 20';
                  }
                  return null;
                },
              ),
              
              const SizedBox(height: 30),
              
              ElevatedButton(
                onPressed: _isLoading ? null : _makePrediction,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade700,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                ),
                child: _isLoading 
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text(
                      'Predict',
                      style: TextStyle(fontSize: 18),
                    ),
              ),
              
              const SizedBox(height: 20),
              
              if (_result.isNotEmpty)
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.green.shade50,
                    border: Border.all(color: Colors.green.shade200),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Prediction Result',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.green.shade800,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        _result,
                        style: const TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _ageController.dispose();
    _yearsController.dispose();
    super.dispose();
  }
}
