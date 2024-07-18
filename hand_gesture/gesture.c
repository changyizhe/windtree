void setup() {
  // Initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

void loop() {
  // Check if data is available to read
  if (Serial.available() > 0) {
    // Read the incoming string
    String command = Serial.readStringUntil('x');

    // Check the command and call the corresponding function
    if (command == "ss001") {
      function1();
    } else if (command == "ss002") {
      function2();
    }
  }
}

void function1() {
  // Function 1: Index Point Up
  // Add your code here
}

void function2() {
  // Function 2: Index Point Down
  // Add your code here
}
