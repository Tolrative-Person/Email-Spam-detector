const display = document.getElementById('display');

function appendValue(value) {
  display.value += value;
}

function clearDisplay() {
  display.value = '';
}
function backspace(){
  console.log(display.value);
  display.value=display.value.substring(0,display.value.length - 1);
  console.log(display.value);
}

function calculate() {
  try {
    let expression = document.getElementById('display').value;
     expression = expression.replace(/sin\(/g, 'Math.sin(');
            expression = expression.replace(/cos\(/g, 'Math.cos(');
            expression = expression.replace(/tan\(/g, 'Math.tan(');
            document.getElementById('display').value = eval(expression);
  } catch {
    display.value = 'Error';
  }
}
