fn main() {
    println!("Hello, world!");

    another_function();
}

fn another_function() {
    println!("Another function.");
}

// Functions
//  - Functions require type annoations for all arguments
fn print_labeled_measurement(value: i32, unit_label: char) {
    println!("The measurement is: {value}{unit_label}");
}

//  - Statement: A line that does not evaluate to anything
//  - Expression: A line that has a returned value
//  - Expressions will NOT end with a semicolon - putting it returns astatement
// let x = (let y = 4) // ALERT: Error because y does not evaluate to anything
let y = {
    let x = 4;
    x + 1  // No semicolon here to return value
}

//  - Expressions must have their return value type defined
fn plus_one(val: i32) -> i32 {
    val + 1
}

fn run() {
    let x = plus_one(5)

    println!("Value of x is: {x}")
}