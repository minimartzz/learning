// ==== Structs ========================================
// - Structs hold multiple related values or different types
// - Dot notation to get the value
// - Can change the value if it's mutable
struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64
}
// Tuple structs
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);
// Unit structs - to implement a trait on some type but don't have the data type yet
struct AlwaysEqual;

fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        email,
        username,
        sign_in_count: 1
    }
}

// ---- Debugging Structs --------------------
#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32
}

#[derive(Debug)]
struct RectangleDebug {
    width: u32,
    height: u32
}

fn area(rectangle: &Rectangle) -> u32 {
    rectangle.width * rectangle.height
}

// ---- Methods --------------------
#[derive(Debug)]
struct RectangleMethod {
    width: u32,
    height: u32,
}

impl RectangleMethod {
    // &self is used to reference the attribtues of the struct
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

struct WithinRectangle {
    width: u32,
    height: u32,
}

impl WithinRectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }

    fn can_hold(&self, other: &WithinRectangle) -> bool {
        self.width > other.width && self.height > other.height
    }
}

// Associated functions are those that take the signature of the original struct
//      and repurpose them to use the attributes of the current struct to redefine
//      themselves
struct AssociatedRectangle {
    width: u32,
    height: u32
}

impl AssociatedRectangle {
    fn square(size: u32) -> Self {
        Self {
            width: size,
            height: size
        }
    }
}

fn main() {
    // Instantiating using a struct
    let user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };

    // Using a previous struct
    // - ..struct1 syntax
    // - If copying a data type that does NOT implement the Copy trait (e.g Strings),
    //   then struct1 will not exist after
    let user2 = User {
        email: String::from("someoneelse@example.com"),
        ..user1
    };

    // Instantiating tuple structs
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
    // - destructuring a tuple struct
    let Point(x, y, z) = origin;

    // Prgram to calculate a rectangle
    let rect1 = Rectangle {
        width: 30,
        height: 50
    };

    println!("The area of the rectange is {}", area(&rect1));
    // :? invokes the debug method which prints the values inside the defined struct
    //     but requires the `derive(Debug)` over the struct definition because by default
    //     it doesn't exist
    println!("The parameters of the rectangle are {rect1:?}");  
    println!("The parameters of the rectangle are {rect1:#?}");  // Formatted 

    // Using !dbg will take ownership unless using a reference
    let scale = 2;
    let rect2 = RectangleDebug {
        width: dbg!(30 * scale),
        height: 50
    };

    dbg!(&rect2);


    let rect3 = RectangleMethod {
        width: 60,
        height: 40
    };
    println!("The area of the rectangle is {}", rect3.area());

    // Using the methods
    let rect4 = WithinRectangle {
        width: 30,
        height: 50,
    };
    let rect5 = WithinRectangle {
        width: 10,
        height: 40,
    };
    let rect6 = WithinRectangle {
        width: 60,
        height: 45,
    };

    println!("Can rect4 hold rect5? {}", rect4.can_hold(&rect5));
    println!("Can rect4 hold rect6? {}", rect4.can_hold(&rect6));

    // Defining with an associated function
    let square = AssociatedRectangle::square(10);
    println!("The width and height of the square is: {0}, {1}", square.width, square.height)
}
