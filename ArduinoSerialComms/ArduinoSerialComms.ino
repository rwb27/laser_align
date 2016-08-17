/*
  Reads an analog input on pin 0, prints the result to the serial monitor.
  Graphical representation is available using serial plotter (Tools > Serial Plotter menu)
  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.
*/

/*
 Consider these factors when using the Arduino as an ADC:
 - What is the sampling rate of the analog signal, and does it satisfy the Nyquist rate (is this even relevant)?
 - Is the scale linear? Does the output require calibration to a given voltage analogue input?
 - Degree of quantisation error?
 - Level of noise?
 - Repeated averaging of ADC values to reduce error.
 - How should the comms request a certain number of measurements?
 */

void voltage_reader(int N, float t, readings[]);

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
// Possible commands that might need to be executed here are: read the serial input, parse it appropriately, 
// take N measurements with a sample rate of R, return bit rate, current model number, firmware info, connected pins, return done when a command is done
// interrupt if needed. Reducing the effects of noise? (With filter?)
void loop() {
  if (Serial.available() > 0){
    // read the input on analog pin 0:
    //int sensorValue = analogRead(A0);
    // print out the value you read:
    //String reading = Serial.readStringUntil(' ');
    //Serial.println(reading);
    //Serial.println(sensorValue);
    //delay(1); // delay in between reads for stability
    int N = 100;
    int readings[N];
    voltage_reader(N, 1000, &readings);
    //Serial.println(readings);
  }
}

// Function to read N measurements with a time delay of t between each measurement and return the array of measurements.
void voltage_reader(int N, float t, readings[]){

  for (int x = 0; x < N; x++){
    int sensorValue = analogRead(A0);
    readings[x] = sensorValue;
    //printf(readings[x]);
    delay(t); // Note t is is milliseconds!
  }
}

