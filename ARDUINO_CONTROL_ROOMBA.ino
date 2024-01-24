#include <SoftwareSerial.h>
// RX is digital pin 10
// (connect to TX of other device - iRobot DB25 pin 2)
// TX is digital pin 11
// (connect to RX of other device - iRobot DB25 pin 1)
#define rxPin 10
#define txPin 11
String str;
int bumpRight,bumpLeft,l,fl,fr,r;
// set up a new software serial port:
SoftwareSerial softSerial = SoftwareSerial(rxPin, txPin);
void setup() {
 delay(2000); // NEEDED!!!! To let the robot initialize

 // define pin modes for software tx, rx pins:
 pinMode(rxPin, INPUT);
 pinMode(txPin, OUTPUT);

 // set the data rate for the SoftwareSerial port, this is the
 // iRobot's default rate of 57600 baud:
 Serial.begin(57600);
 softSerial.begin(57600);
 softSerial.write(128); // This command starts the OI. You must
 // always send the Start command before
// sending any other commands to the OI
 softSerial.write(131); // safe mode
}
void loop()
{
 if (Serial.available())
  {
    str = Serial.readStringUntil('\n');
    if(str == "FORWARD")
      goForward();
    if(str == "BACKWARD")
      goBackward();
    if(str == "TURN_LEFT")
      goLeft();
    if(str == "TURN_RIGHT")
      goRight();
  } 
  checkSensors() ;
}

void goForward() 
{
    softSerial.write(137); 
    softSerial.write((byte)0);
    softSerial.write((byte)200);
    softSerial.write((byte)128); 
    softSerial.write((byte)0);
    Serial.println("Go Forward"); 
}

void goBackward()
{
    softSerial.write(137);
    softSerial.write((byte)255);
    softSerial.write((byte)56);
    softSerial.write((byte)128);
    softSerial.write((byte)0);
    Serial.println("Go Backward"); 
}
void goLeft()
{
    softSerial.write(137);
    softSerial.write((byte)0);
    softSerial.write((byte)200);
    softSerial.write((byte)0);
    softSerial.write((byte)1);
    Serial.println("Turn Left"); 
}
void goRight()
{
    softSerial.write(137);
    softSerial.write((byte)0);
    softSerial.write((byte)200);
    softSerial.write((byte)255);
    softSerial.write((byte)255);
    Serial.println("Turn Right"); 
}
void checkSensors() 
{
 char sensorbytes[10]; 

 softSerial.write((byte)142); 
 softSerial.write((byte)1); 
 delay(64);
 char i = 0;
 while (i < 10) {
 sensorbytes[i++] = 0;
 }
 i = 0;
 while(softSerial.available()) {
 int c = softSerial.read();
 sensorbytes[i++] = c;
 }

 bumpRight = sensorbytes[0] & 0x01;
 bumpLeft = sensorbytes[0] & 0x02;
 l=sensorbytes[2];
 fl=sensorbytes[3];
 fr=sensorbytes[4];
 r=sensorbytes[5];

 Serial.print(bumpRight);
 Serial.print(" ");
 Serial.print(bumpLeft);
 Serial.print(" ");
 Serial.print(l);
 Serial.print(" ");
 Serial.print(fl);
 Serial.print(" ");
 Serial.print(fr);
 Serial.print(" ");
 Serial.println(r);
}
