// ==== IF Statement ====================
// - All if expressions must evaluation to bool. Any other data type will retun an error
fn if_statement(val: i32) {
    if val == 5 {
        println!("The value of the number is 5");
    } else if val == 6 {
        println!("The value of the number is 6");
    } else {
        println!("The value of the number is NOT 5");
    }
}

// - Ternary statements
fn ternary(val: i32) {
    let condition = true;
    let x = if condition { 5 } else { 6 };
    println!("The value of x is: {x}");
}


// ==== LOOP Statement ====================
// - 3 types of "loops": loop, for, while
fn loop_example() {
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    }
    println!("The final value of counter is: {counter}")
}

// - Loop labels help to disambiguate nested loops
// - loop labels are marked with a single quote at the start 'counting_up
fn nested_loop() {
    let mut count = 0;
    'counting_up loop {
        println!("count = {count}")
        let mut remainder = 10;

        loop {
            println!("remainder = {remainder}")
            if remainder == 9 {
                break;
            }
            if count == 2 {
                break 'counting_up;
            }

            remainder -= 1;
        }

        count += 1;
    }
    println!("End of count: {count}");
}


// ==== WHILE Statement ====================
fn while_example() {
    let a = [10, 20, 30, 40, 50];
    let mut index = 0;

    while index < 5 {
        println!("the value is: {}", a[index]);

        index += 1;
    }
}


// ==== FOR Statement ====================
fn for_example() {
    let a = [10, 20, 30, 40, 50];

    for element in a {
        println!("the value is: {element}");
    }
}

// FOR statement with Range

fn for_with_range() {
    for number in (1..4).rev() {
        println!({number}!)
    }
    println!("LIFTOFF!")
}