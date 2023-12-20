#include <ESP8266WiFi.h>
#include <DHT.h>
#include <ThingSpeak.h>

const char *ssid =  "realme3Pro";  
const char *pass =  "7603863634";

//const char *ssid =  "Sam";  
//const char *pass =  "samssree";


#define DHTPIN 5            //pin where the dht11 is connected
#define RELAYPIN 15

DHT dht(DHTPIN, DHT11);

WiFiClient client;

const char *myWriteAPIKey = "UHR91ZZ5KRL2QBD4";
unsigned long myChannelNumber = 2266425;

 // Decision returns to this channel
unsigned long returnChannelNumber = 2288225;
const char *returnReadAPIKey = "025ZCWEFOV7549DG";

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  WiFi.begin(ssid, pass);
  while(WiFi.status() != WL_CONNECTED)
  {
    delay(200);
    Serial.print("..");
  }
  Serial.println();
  Serial.println("NodeMCU is connected!");
  Serial.println(WiFi.localIP());
  dht.begin();
  ThingSpeak.begin(client);
  pinMode(RELAYPIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  const int SensorPin = A0;
  
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  int soil_moisture = 0;
  int soil_moist_percent = 0;

  soil_moisture = analogRead(SensorPin);

  soil_moisture = soil_moisture / 2;

  Serial.println("Moisture: " + (String) soil_moisture);

  Serial.println("Temperature: " + (String) t);
  Serial.println("Humidity: " + (String) h);

  ThingSpeak.setField(1, t);
  ThingSpeak.setField(2, h);
  ThingSpeak.setField(3, soil_moisture);

  int x = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

  if (x == 200){
    Serial.println("Sent Successfully");
  }
  delay(15000);

  int decision = ThingSpeak.readIntField(returnChannelNumber, 1, returnReadAPIKey);
  
  if(decision == 0){
    Serial.println("Irrigate the land");
    Serial.println("");
    digitalWrite(RELAYPIN, LOW);
    delay(5000);
  }
  else{
    Serial.println("No irrigation is required");
    digitalWrite(RELAYPIN, HIGH);
    delay(5000);
  }

  delay(15000);
}
