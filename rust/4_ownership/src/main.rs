// ==== Ownership ========================================
// - Ownership is Rusts way of managing memory in a program: A set of rules the compiler checks.
//      if any of them are violated, the program won't compile
// - Rules:
//      1. Each value in Rust has an owner
//      2. There can only be one owner at a time
//      3. When the owner goes out of scope, the value is dropped
// - Previous data types are immutable, so there is a need for more flexible data types

// ---- Stack vs. Heap --------------------
// - 2 different data structures Rust uses to manage memory
// - Stack: LIFO, all data on the stack must have a known fixed size
// - Heap: Allocating data of ANY size into any free space and returning the pointer
//      * Pointers in the heap can be stored in the stack
// - Pushing and retrieving data from the Heap is slower, but more flexible

// ==== String Data Type ====================
// - Allows us to muta
fn main() {
    let mut s = String::from("hello");
    s.push_str(", world!");
    println!("{s}");

    double_free_error();

    deep_copy();
    
    copy_example();

    func_scope();
}

// - When "copying" a variable on the heap, only the pointer is copied, underlying data is the same
// - Because Rust calls the `drop` function to clean up memory at the end of the scope, if trying to
//      clean up 2 variables (s1 and s2) from memory it will try to do it twice, but with only 1 set of
//      underlying data -- double free error
// - Declaring s2 will make s1 invalid
// - Rust never makes a deep copy, any vopyinh can be considered inexpensive
fn double_free_error() {
    let s1 = String::from("hello");
    let s2 = s1;
    println!("{s2}, world!");
}

// Deep copies are expensive because ALL data in the stack is being replicated
fn deep_copy() {
    let s1 = String::from("hello");
    let s2 = s1.clone();
    println!("s1 = {s1}, s2 = {s2}");
}

// `Copy` is a trait that allows variables to be copied when being reassigned automatically
// Below example is implemented automatically for the integer data type
// Happens because the memory capacity used is small, so it's trivial to copy over during reassignment
// Types that implement Copy by default:
//      - integers, floats, booleans, characters
//      - tuples that have types that implement Copy
fn copy_example() {
    let x = 5;
    let y = x;
    println!("x = {x}, y = {y}");
}

// ==== Function Scopes ====================
fn func_scope() {
    let s = String::from("hello");  // s comes into scope
    println!("'{s}' is still accessible here");

    takes_ownership(s);             // s's value moves into the function...
                                    // ... and so is no longer valid here
    // println!("'{s}' is no longer accessible here");

    let x = 5;                      // x comes into scope
    println!("'{x}' is still accessible here");

    makes_copy(x);                  // Because i32 implements the Copy trait,
                                    // x does NOT move into the function,
                                    // so it's okay to use x afterward.
    println!("'{x}' is still accessible here");

} // Here, x goes out of scope, then s. However, because s's value was moved,
  // nothing special happens.

fn takes_ownership(some_string: String) { // some_string comes into scope
    println!("{some_string}");
} // Here, some_string goes out of scope and `drop` is called. The backing
  // memory is freed.

fn makes_copy(some_integer: i32) { // some_integer comes into scope
    println!("{some_integer}");
} // Here, some_integer goes out of scope. Nothing special happens.