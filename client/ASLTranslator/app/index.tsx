import React, { useState, useEffect, useRef } from "react";
import {
  Text,
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  FlatList,
  Image,
} from "react-native";
import { io } from "socket.io-client";
import * as BleManager from "react-native-ble-manager";
import { NativeEventEmitter, NativeModules } from "react-native";

// Harcoded URL for the server
const SERVER_URL = "http://10.1.10.225:8000";

export default function Index() {
  const [isConnected, setIsConnected] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  // New States for translation
  const [currentTranslation, setCurrentTranslation] = useState("");
  const [translationHistory, setTranslationHistory] = useState([]);
  const [socket, setSocket] = useState(null);
  const [landmarkImage, setLandmarkImage] = useState(null);

  // Initialize WebSocket connection
  useEffect(() => {
    // Create socket instance
    const newSocket = io(SERVER_URL);

    // Socket event handlers
    newSocket.on("connect", () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      setStatusMessage((prev) => prev + "\nWebSocket connected");
    });

    newSocket.on("disconnect", () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      setStatusMessage((prev) => prev + "\nWebSocket disconnected");
    });

    newSocket.on("translation_update", (data) => {
      console.log("Translation received:", data);

      // Update current translation
      setCurrentTranslation(data.sign);

      // Add to history
      setTranslationHistory((prev) => {
        const newHistory = [...prev, data];
        // Keep only the last 10 translations
        if (newHistory.length > 10) {
          return newHistory.slice(newHistory.length - 10);
        }
        return newHistory;
      });
    });

    newSocket.on("translation_status", (data) => {
      console.log("Translation status:", data);
      if (data.status === "connected") {
        setIsTranslating(true);
      } else if (data.status === "stopped") {
        setIsTranslating(false);
      }
      setStatusMessage(data.message);
    });

    newSocket.on("translation_error", (data) => {
      console.log("Translation error:", data);
      setStatusMessage(`Error: ${data.message}`);
      setIsTranslating(false);
    });

    newSocket.on("landmark_visualization", (data) => {
      console.log("Landmark visualization received");
      setLandmarkImage(data.image);
    });

    // Save socket instance
    setSocket(newSocket);

    // Cleanup on unmount
    return () => {
      newSocket.disconnect();
    };
  }, []);

  const checkServerStatus = async () => {
    setStatusMessage("Checking server connection...");

    try {
      const response = await fetch(`${SERVER_URL}/status`, {
        method: "GET",
      });

      if (response.ok) {
        const data = await response.json();
        setIsConnected(true);
        setIsTranslating(data.running || false);
        setStatusMessage(
          `Connected to server! ${
            isTranslating
              ? "Translation is active."
              : "Server is ready. Translation inactive."
          }`
        );
      } else {
        setIsConnected(false);
        setStatusMessage(
          `Failed to connect to server: HTTP ${response.status}`
        );
      }
    } catch (error) {
      setIsConnected(false);
      setStatusMessage(`Failed to connect to server: ${error.message}`);
    }
  };

  const startTranslation = async () => {
    setStatusMessage("Starting ASL translation...");
    // Clear previous translations
    setCurrentTranslation("");
    setTranslationHistory([]);

    try {
      const response = await fetch(`${SERVER_URL}/start_translation`, {
        method: "POST",
      });

      if (response.ok) {
        setIsTranslating(true);
        setStatusMessage("ASL translation started successfully!");
      } else {
        setStatusMessage(
          `Failed to start translation: HTTP ${response.status}`
        );
      }
    } catch (error) {
      setStatusMessage(`Failed to start translation: ${error.message}`);
    }
  };

  const stopTranslation = async () => {
    setStatusMessage("Stopping ASL translation...");

    try {
      const response = await fetch(`${SERVER_URL}/stop_translation`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        setIsTranslating(false);
        setStatusMessage("ASL translation stopped successfully!");
      } else {
        setStatusMessage(`Failed to stop translation: HTTP ${response.status}`);
      }
    } catch (error) {
      setStatusMessage(`Failed to stop translation: ${error.message}`);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ASL Translator</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Connection</Text>
        <TouchableOpacity style={styles.button} onPress={checkServerStatus}>
          <Text style={styles.buttonText}>Check Connection</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Translation Control</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.startButton,
              !isConnected && styles.disabledButton,
            ]}
            onPress={startTranslation}
            disabled={!isConnected}
          >
            <Text style={styles.buttonText}>Start Translation</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.actionButton,
              styles.stopButton,
              (!isConnected || !isTranslating) && styles.disabledButton,
            ]}
            onPress={stopTranslation}
            disabled={!isConnected || !isTranslating}
          >
            <Text style={styles.buttonText}>Stop Translation</Text>
          </TouchableOpacity>
        </View>
      </View>

      {isTranslating && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Live Translation</Text>
          <Text style={styles.currentSign}>
            {currentTranslation || "Waiting..."}
          </Text>
          <Text style={styles.historyTitle}>Recent:</Text>
          <FlatList
            data={translationHistory.slice().reverse()}
            keyExtractor={(_, i) => i.toString()}
            renderItem={({ item }) => (
              <View style={styles.historyItem}>
                <Text style={styles.historySign}>{item.sign}</Text>
                <Text style={styles.historyConfidence}>
                  {(item.confidence * 100).toFixed(0)}%
                </Text>
              </View>
            )}
            style={styles.historyList}
          />
        </View>
      )}

      {/* {isTranslating && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Landmark Visualization</Text>
          {landmarkImage ? (
            <Image
              source={{ uri: landmarkImage }}
              style={styles.landmarkImage}
              resizeMode="contain"
            />
          ) : (
            <Text>Waiting for visualization...</Text>
          )}
        </View>
      )} */}

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Status</Text>
        <Text>{statusMessage}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: "#f5f5f5" },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    textAlign: "center",
    marginTop: 40,
    marginBottom: 20,
  },
  card: {
    backgroundColor: "white",
    borderRadius: 8,
    padding: 16,
    marginBottom: 16,
  },
  cardTitle: { fontSize: 18, fontWeight: "bold", marginBottom: 12 },
  button: {
    backgroundColor: "#007bff",
    paddingVertical: 10,
    borderRadius: 4,
    alignItems: "center",
  },
  buttonText: { color: "white", fontWeight: "500" },
  buttonRow: { flexDirection: "row", justifyContent: "space-between" },
  actionButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 4,
    alignItems: "center",
  },
  startButton: { backgroundColor: "#28a745", marginRight: 8 },
  stopButton: { backgroundColor: "#dc3545" },
  disabledButton: { backgroundColor: "#cccccc", opacity: 0.6 },
  currentSign: {
    fontSize: 32,
    fontWeight: "bold",
    textAlign: "center",
    marginVertical: 10,
    color: "#007bff",
  },
  historyTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginTop: 16,
    marginBottom: 8,
  },
  historyList: { maxHeight: 150 },
  historyItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#eee",
  },
  historySign: { fontSize: 16 },
  historyConfidence: { fontSize: 14, color: "#666" },
  landmarkImage: {
    width: "100%",
    height: 200,
    marginVertical: 10,
    borderRadius: 8,
  },
});
