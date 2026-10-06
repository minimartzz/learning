fn main() {
    // 1. Immutable and Mutable variables
    let x = 5;
    let mut z = 5;
    println!("The value of x is: {x}");
    // x = 6;  // ALERT: Error because value is not mutable
    z = 6;
    println!("The value of z is: {z}");

    // 2. Constants
    //   - Constants are not mutable at all
    //   - Constants can be declared in any scope
    //   - Constant expression; not a result of a value
    //   - USE_ALL_UPPERCASE
    //   - Limited number of operations when using const that the compiler evaluates
    const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
    println!("The constant value: {THREE_HOURS_IN_SECONDS}");

    // 3. Shadowing
    //   - Override a previously declared variable
    //   - Defines the variable with the same name in local scope
    //   - Allows changing of data type
    //   - Effectively creating a new variable with let
    let y = 2;
    let y = 2 + 1;

    {
        let y = y * 2; // Returns: 6
        println!("The value of y: {y}");
    }

    println!("The value of y: {y}"); // Returns: 3

    let spaces = "   ";
    let spaces = spaces.len();
    println!("The number of spaces: {spaces}");
}
