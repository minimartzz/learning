fn data_types() {
    // SCALARS
    // 1. Integers
    //   - Values perform a wraparound if it goes outside the max/ min data type
    //   - Test by compiling. Running with --release will not return panic!
    let integer: u32 = 500;
    println!("Integer type: {integer}");

    // 2. Floats
    //   - f32 and f64 only. All are signed. Default f64
    let float: f64 = 500.432;

    // 3. Boolean
    let boolean: bool = false;

    // 4. Character
    //   - Must be single quotes
    let character: char = 'z';

    // COMPOUND
    // 1. Tuples
    //   - Each line item corresponds to their type position
    //   - Items are accessed by .<pos>
    //   - Can also be mutable with `mut`
    let tup: (i32, f64, u8) = (500, 6.4, 1);
    let (x, y, z) = tup;

    println!("The value of y is: {y}");

    let five_hundred = tup.0;
    let size_point_four = tup.1;
    let one = tup.2;

    // 2. Array
    //   - EVERY ELEMENT MUST BE THE SAME TYPE
    //   - Fixed length
    //   - Items are accessed by [pos]
    let months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let a: [i32; 5] = [1, 2, 3, 4, 5];  // Defines the exact number of items in array
    let b: [3; 5];  // Initialise the array with all 3s, 5 times

    let first = a[0];
    let second = a[1];
}
