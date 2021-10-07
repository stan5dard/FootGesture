#ifndef SERIAL_RATE
#define SERIAL_RATE         230400
#endif

#ifndef cbi   // clear bit
#define cbi(sfr, bit) (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi   // set bit
#define sbi(sfr, bit) (_SFR_BYTE(sfr) |= _BV(bit))
#endif


const int selectRowPins[5] = {2, 3, 4, 5, 6};     // S0~2, S1~3, S2~4
const int selectReadPins[5] = {8, 9, 10, 11, 12}; // S0~9, S1~10
const int ledPin = 13;
uint16_t a0val = 0;
uint16_t numAvg = 0;
const int zInput = A0;


void setup() {
  sbi(ADCSRA, ADPS2); // ADCSRA |= B00000100; ADC prescaler is 16. 16MHz/16 = 1MHz
  cbi(ADCSRA, ADPS1);
  cbi(ADCSRA, ADPS0);

  Serial.begin(SERIAL_RATE, SERIAL_8N1);

  for (int i = 0; i < 5; i++) {
    pinMode(selectRowPins[i], OUTPUT);
    pinMode(selectReadPins[i], OUTPUT);
    digitalWrite(selectRowPins[i], LOW);
    digitalWrite(selectReadPins[i], LOW);
  }
  
  pinMode(ledPin, OUTPUT);
  pinMode(zInput, INPUT);
}

void loop() {
  switch (readData()) {
    case 97:
      digitalWrite(ledPin, HIGH);
      scanArray();
      digitalWrite(ledPin, LOW);
      break;
  }
  
  delay(5);
}

char readData() {
  Serial.println("w");
  while(1) {
    if(Serial.available() > 0) {
      short val = (short) Serial.read();
      // Serial.println(val);
      return val;
    }
  }
}

void scanArray() {
  numAvg = 3;   
  
  for (int pin = 0; pin <= 31; pin++) {
    selectMuxPin(pin);
        
    for(int readPin = 0; readPin <= 31; readPin++) {
      selectReadSwitch(readPin);
      delayMicroseconds(20);
      a0val = 0;
      for(int avg = 0; avg < numAvg; avg++) {
          a0val = a0val + analogRead(zInput);
      }
      
      a0val = a0val / numAvg;

      // Serial.println(a0val);

      // 10 bit version 
      Serial.write((uint8_t)(a0val>>5));  
      Serial.write((uint8_t)(a0val & 0x1F));
    }
  }     
}

void selectMuxPin(byte pin) {
  if (pin > 31) return; // Exit if pin is out of scope
      
  for (int i = 0; i < 5; i++) {
    if (pin & (1<<i))
      digitalWrite(selectRowPins[i], HIGH);
    else
      digitalWrite(selectRowPins[i], LOW);
  }
}

void selectReadSwitch(byte pin) {
  if (pin > 31) return; // Exit if pin is out of scope
      
  for (int i = 0; i < 5; i++) {
    if (pin & (1<<i))
      digitalWrite(selectReadPins[i], HIGH);
    else
      digitalWrite(selectReadPins[i], LOW);
  }
}
