/*
  Reads an analog input on pin 0 when fed a single character and returns a
  single reading.
*/

void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

void loop() {
  while (Serial.available() > 0) {
    // read the input on analog pin 0:
    Serial.read();

    int sensorValue = analogRead(A0);
    // delay in between reads for stability
    delay(1);
    // print out the value you read:
    Serial.println(sensorValue);
  }
}
