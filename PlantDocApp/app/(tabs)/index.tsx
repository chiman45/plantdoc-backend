import { useState } from 'react';
import {
  StyleSheet, Text, View, TouchableOpacity,
  Image, ScrollView, ActivityIndicator, Alert
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = 'https://plantdoc-backend-5vll.onrender.com';

export default function HomeScreen() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    let res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaType.Images,
      quality: 1,
    });
    if (!res.canceled) {
      setImage(res.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    let res = await ImagePicker.launchCameraAsync({
      quality: 1,
    });
    if (!res.canceled) {
      setImage(res.assets[0].uri);
      setResult(null);
    }
  };

  const predict = async () => {
    if (!image) return Alert.alert('Please select an image first!');
    setLoading(true);
    try {
      const formData = new FormData();
      const isWeb = typeof document !== 'undefined';
      if (isWeb) {
        const imgResponse = await fetch(image);
        const blob = await imgResponse.blob();
        formData.append('file', blob, 'leaf.jpg');
      } else {
        formData.append('file', { uri: image, type: 'image/jpeg', name: 'leaf.jpg' } as any);
      }
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (e) {
      Alert.alert('Error', 'Could not connect to server!');
    }
    setLoading(false);
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>🌿 PlantDoc</Text>
        <Text style={styles.headerSub}>Plant Disease Detection</Text>
      </View>

      <ScrollView style={styles.content}>
        <View style={styles.uploadBox}>
          {image ? (
            <Image source={{ uri: image }} style={styles.leafImage} />
          ) : (
            <Text style={styles.uploadText}>No image selected</Text>
          )}
        </View>

        <View style={styles.btnRow}>
          <TouchableOpacity style={styles.btnGreen} onPress={takePhoto}>
            <Text style={styles.btnText}>📷 Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.btnGreen} onPress={pickImage}>
            <Text style={styles.btnText}>🖼️ Gallery</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.predictBtn} onPress={predict}>
          <Text style={styles.predictText}>🔍 Detect Disease</Text>
        </TouchableOpacity>

        {loading && <ActivityIndicator size="large" color="#4ade80" style={{marginTop: 20}}/>}


       {result && (
  <View style={styles.resultCard}>
    <Text style={styles.resultTitle}>Detection Result</Text>
    {result.confidence < 40 ? (
      <View style={styles.warningBox}>
        <Text style={styles.warningText}>⚠️ Low Confidence!</Text>
        <Text style={styles.warningSubText}>Please upload a clear plant leaf image for accurate results.</Text>
      </View>
    ) : (
      <>
        <Text style={styles.diseaseName}>{result.disease}</Text>
        <Text style={styles.confidence}>Confidence: {result.confidence}%</Text>
      </>
    )}
  </View>
)}

      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0d1117' },
  header: { backgroundColor: '#0a2a1a', padding: 40, alignItems: 'center' },
  headerText: { color: '#4ade80', fontSize: 28, fontWeight: 'bold' },
  headerSub: { color: '#6b7280', fontSize: 14, marginTop: 4 },
  content: { padding: 20 },
  uploadBox: {
    backgroundColor: '#111827', borderRadius: 16,
    borderWidth: 2, borderColor: '#2d4a3e', borderStyle: 'dashed',
    height: 220, alignItems: 'center', justifyContent: 'center', marginBottom: 16
  },
  leafImage: { width: '100%', height: '100%', borderRadius: 16 },
  uploadText: { color: '#6b7280', fontSize: 16 },
  btnRow: { flexDirection: 'row', gap: 12, marginBottom: 16 },
  btnGreen: {
    flex: 1, backgroundColor: '#1e3a2f', padding: 14,
    borderRadius: 12, alignItems: 'center'
  },
  btnText: { color: '#4ade80', fontSize: 15, fontWeight: '600' },
  predictBtn: {
    backgroundColor: '#4ade80', padding: 16,
    borderRadius: 12, alignItems: 'center', marginBottom: 20
  },
  predictText: { color: '#0a2a1a', fontSize: 16, fontWeight: 'bold' },
  resultCard: {
    backgroundColor: '#111827', borderRadius: 16,
    padding: 20, borderLeftWidth: 4, borderLeftColor: '#4ade80'
  },
  resultTitle: { color: '#6b7280', fontSize: 13, marginBottom: 8 },
  diseaseName: { color: '#e5e7eb', fontSize: 20, fontWeight: 'bold', marginBottom: 4 },
  confidence: { color: '#4ade80', fontSize: 14 },

   warningBox: {
  backgroundColor: '#3a1e1e',
  borderRadius: 10,
  padding: 12,
},
warningText: {
  color: '#f87171',
  fontSize: 16,
  fontWeight: 'bold',
  marginBottom: 4,
},
warningSubText: {
  color: '#fca5a5',
  fontSize: 13,
  lineHeight: 18,
},
});