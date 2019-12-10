#include <Wire.h>

#define VERBOSE false
#define DEBUG false
#define LOST_TRACKING 69

const int ledPin=13;
uint8_t sys, gyro = 0, accel = 0, mag = 0;

const unsigned int DATA_LEN = 200;
char data1[DATA_LEN];
char data2[DATA_LEN];
unsigned int ind1, ind2, check1, check2;
boolean reading1, reading2;

float d1[2] = {}; float d2[2] = {}; float d3[2] = {}; float d4[2] = {}; int correspondence_count = 0;
float b1[2] = {}; float b2[2] = {}; float b3[2] = {}; float b4[2] = {};

void setup() {
  /* setup serial & UART */

  Serial.begin(115200); Serial1.begin(115200); Serial2.begin(115200);

  setupFlag();
  Serial.println("Waiting for UART communication to activate.");

  while (!(Serial1 && Serial2))  {
    ; // wait for serial to connect
  }

  setupFlag();
  Serial.println("UART connection established between Teensies. Setup complete. Now running.");
  ind1 = 0; ind2 = 0; check1 = 0; check2 = 0;
  reading1 = false; reading2 = false;
}

void loop() {
  // build up poses while UART ports are available
  while (Serial1.available() || Serial2.available()) {
    if (Serial1.available()) {
      readSerial1();
    }

    if (Serial2.available()) {
      readSerial2();
    }
  }
}

// MARK: UART comms

void printAngleData() {
  if (correspondence_count == 0b1111) {
    // Serial.printf("[(%f, %f), (%f, %f), (%f, %f), (%f, %f)]\n", d1[0], d1[1], d2[0], d2[1], d3[0], d3[1], d4[0], d4[1]);
    Serial.printf("[[(%f, %f), (%f, %f), (%f, %f), (%f, %f)], [(%f, %f), (%f, %f), (%f, %f), (%f, %f)]]\n", d1[0], d1[1], d2[0], d2[1], d3[0], d3[1], d4[0], d4[1], b1[0], b1[1], b2[0], b2[1], b3[0], b3[1], b4[0], b4[1]);
    correspondence_count = 0;
  }
}

String getValue(String data, char separator, int index) {
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void readSerial1() {
  if (ind1 < DATA_LEN-1) {
    char inChar = Serial1.read(); // Read a character
    if (inChar == '+') {
      reading1 = true;
      ind1 = 0;
    }

    if (reading1) {
      data1[ind1] = inChar;
      ind1 += 1;
      if (inChar == '\n') {
        reading1 = false;
        ind1 -= 1; // remove endline
        data1[ind1] = '\0'; // null terminate that bih

        String myString = String(data1);
        if (check1 == 6) { // check if tracking
          // send over angle data
          if (VERBOSE) {
            Serial.print("T1:\t");
            Serial.print(myString);
            Serial.println();
          } else {
            if (getValue(myString, '\t', 0) == "+ANG0") {
              String d1_0 = getValue(myString, '\t', 3);
              String d1_1 = getValue(myString, '\t', 4);
              String b1_0 = getValue(myString, '\t', 5);
              String b1_1 = getValue(myString, '\t', 6);
              if (d1_0 != "" && d1_1 != "") {
                d1[0] = d1_0.toFloat(); d1[1] = d1_1.toFloat();
              } else {
                d1[0] = LOST_TRACKING; d1[1] = LOST_TRACKING;
              }

              if (b1_0 != "" && b1_1 != "") {
                b1[0] = b1_0.toFloat(); b1[1] = b1_1.toFloat();
              } else {
                b1[0] = LOST_TRACKING; b1[1] = LOST_TRACKING;
              }
              correspondence_count |= 0b1;
            } else {
              String d2_0 = getValue(myString, '\t', 3);
              String d2_1 = getValue(myString, '\t', 4);
              String b2_0 = getValue(myString, '\t', 5);
              String b2_1 = getValue(myString, '\t', 6);
              if (d2_0 != "" && d2_1 != "") {
                d2[0] = d2_0.toFloat(); d2[1] = d2_1.toFloat();
              } else {
                d2[0] = LOST_TRACKING; d2[1] = LOST_TRACKING;
              }

              if (b2_0 != "" && b2_1 != "") {
                b2[0] = b2_0.toFloat(); b2[1] = b2_1.toFloat();
              } else {
                b2[0] = LOST_TRACKING; b2[1] = LOST_TRACKING;
              }
              correspondence_count |= 0b10;
            }
            printAngleData();
          }
        }
        reading1 = false; ind1 = 0; check1 = 0; // reset params
      } else if (inChar == '\t') { // delimiter
        check1 += 1;
      }
    }
  } else {
    errorFlag();
    Serial.println("Hit end of data array 1. Try extending capacity."); reading1 = false; ind1 = 0; check1 = 0;
  }
}

void readSerial2() {
  if (ind2 < DATA_LEN-1) {
    char inChar = Serial2.read(); // Read a character
    if (inChar == '+') {
      reading2 = true;
      ind2 = 0;
    }

    if (reading2) {
      data2[ind2] = inChar;
      ind2 += 1;
      if (inChar == '\n') {
        reading2 = false;
        ind2 -= 1; // remove endline
        data2[ind2] = '\0'; // null terminate that bih

        String myString = String(data2);
        if (check2 == 6) { // check if tracking
          // send over angle data
          if (VERBOSE) {
            Serial.print("T2:\t");
            Serial.print(myString);
            Serial.println();
          } else {
            if (getValue(myString, '\t', 0) == "+ANG0") { // TODO: change cases for 2 LH, add to b_mat
              String d3_0 = getValue(myString, '\t', 3);
              String d3_1 = getValue(myString, '\t', 4);
              String b3_0 = getValue(myString, '\t', 5);
              String b3_1 = getValue(myString, '\t', 6);
              if (d3_0 != "" && d3_1 != "") {
                d3[0] = d3_0.toFloat(); d3[1] = d3_1.toFloat();
              } else {
                d3[0] = LOST_TRACKING; d3[1] = LOST_TRACKING;
              }

              if (b3_0 != "" && b3_1 != "") {
                b3[0] = b3_0.toFloat(); b3[1] = b3_1.toFloat();
              } else {
                b3[0] = LOST_TRACKING; b3[1] = LOST_TRACKING;
              }
              correspondence_count |= 0b100;
            } else {
              String d4_0 = getValue(myString, '\t', 3);
              String d4_1 = getValue(myString, '\t', 4);
              String b4_0 = getValue(myString, '\t', 5);
              String b4_1 = getValue(myString, '\t', 6);
              if (d4_0 != "" && d4_1 != "") {
                d4[0] = d4_0.toFloat(); d4[1] = d4_1.toFloat();
              } else {
                d4[0] = LOST_TRACKING; d4[1] = LOST_TRACKING;
              }

              if (b4_0 != "" && b4_1 != "") {
                b4[0] = b4_0.toFloat(); b4[1] = b4_1.toFloat();
              } else {
                b4[0] = LOST_TRACKING; b4[1] = LOST_TRACKING;
              }
              correspondence_count |= 0b1000;
            }
            printAngleData();
          }
        }
        reading2 = false; ind2 = 0; check2 = 0; // reset params
      } else if (inChar == '\t') { // delimiter
        check2 += 1;
      }
    }
  } else {
    errorFlag();
    Serial.println("Hit end of data array 2. Try extending capacity."); reading2 = false; ind2 = 0; check2 = 0;
  }
}

// MARK: Serial flags & Utils

void dataFlag() {
  Serial.print("Data:\n");
}

void setupFlag() {
  Serial.print("Setup:\t");
}

void errorFlag() {
  Serial.print("Error:\t");
}
