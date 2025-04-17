import React, { useState } from "react";
import {
  Text,
  View,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Clipboard,
} from "react-native";

// Harcoded URL for the server
const SERVER_URL = "http://172.27.230.40:8000";

export default function Index() {
  const [streamUrl, setStreamUrl] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

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
              : "Translation is not active."
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
    if (!streamUrl) {
      setStatusMessage("Please enter an Instagram livestream URL");
      return;
    }

    setStatusMessage("Starting ASL translation...");

    try {
      const response = await fetch(`${SERVER_URL}/start_translation`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ stream_url: streamUrl }),
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

  const pasteFromClipboard = async () => {
    try {
      const content = await Clipboard.getString();
      setStreamUrl(content);
    } catch (error) {
      setStatusMessage("Unable to paste from clipboard");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ASL Translator</Text>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Connection Status</Text>
        <View style={styles.statusHeader}>
          <Text style={styles.serverText}>Server: {SERVER_URL}</Text>
          <View
            style={[
              styles.statusIndicator,
              isConnected
                ? isTranslating
                  ? styles.statusGreen
                  : styles.statusOrange
                : styles.statusRed,
            ]}
          />
        </View>
        <TouchableOpacity
          style={styles.connectButton}
          onPress={checkServerStatus}
        >
          <Text style={styles.buttonText}>Check Connection</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Instagram Livestream</Text>
        <View style={styles.inputGroup}>
          <TextInput
            style={styles.input}
            value={streamUrl}
            onChangeText={setStreamUrl}
            placeholder="https://www.instagram.com/username/live/"
          />
          <TouchableOpacity style={styles.button} onPress={pasteFromClipboard}>
            <Text style={styles.buttonText}>Paste</Text>
          </TouchableOpacity>
        </View>
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

      <View style={styles.card}>
        <Text style={styles.cardTitle}>Status</Text>
        <Text>{statusMessage}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: "#f5f5f5",
  },
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
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 12,
  },
  statusHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  serverText: {
    fontSize: 14,
  },
  connectButton: {
    backgroundColor: "#007bff",
    paddingVertical: 10,
    borderRadius: 4,
    alignItems: "center",
  },
  inputGroup: {
    flexDirection: "row",
    marginBottom: 12,
  },
  input: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 4,
    padding: 8,
    marginRight: 8,
  },
  button: {
    backgroundColor: "#007bff",
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 4,
    justifyContent: "center",
  },
  buttonText: {
    color: "white",
    fontWeight: "500",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  actionButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 4,
    alignItems: "center",
  },
  startButton: {
    backgroundColor: "#28a745",
    marginRight: 8,
  },
  stopButton: {
    backgroundColor: "#dc3545",
  },
  disabledButton: {
    backgroundColor: "#cccccc",
    opacity: 0.6,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  statusRed: {
    backgroundColor: "red",
  },
  statusOrange: {
    backgroundColor: "orange",
  },
  statusGreen: {
    backgroundColor: "green",
  },
});
