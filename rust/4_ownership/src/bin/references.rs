// ==== References ========================================
// - References prevent ownership from being taken into another scope, but allows
//      that scope to access the value through a pointer
// - Marked by an &
// - Opposite is dereferencing marked by *
fn reference_example() {
    let s = String::from("some value");

    let y = count_string(&s);

    println!("The string {s} has length {y}");

}

fn count_string(some_string: &String) -> usize {  // some_string is a reference to a String
    some_string.len()
} // Here, some_string goes out of scope. But because some_string does not have ownership of what
  // it refers to, the String is not dropped

// ---- Mutable References --------------------
// ALERT: References are not mutable unless EXPLICITLY defined as a mutable reference
// - If you have a mutable reference, you can have no other references to that value except for one
fn func_for_mutable_reference(s: &mut String) {
    s.push_str(", world");
}

fn mutable_reference_rules {
    // mutable reference
    let mut x = String::from("Martin's");
    func_for_mutable_reference(&mut x);

    // Cannot create more than 1 immutable reference
    let a1 = &mut x;  // This is ok
    let a2 = &mut x;  // This will fail

    // Cannot mix immutable and mutable references
    let b1 = &x;  // This is ok
    let b2 = &x;  // This is ok
    let b3 = &mut x;  // This will fail - because mutable cannot be assigned after an immutable

    // Once the immutable reference is used, mutable reference can be defined
    let mut s = String::from("hello");

    let r1 = &s; // no problem
    let r2 = &s; // no problem
    println!("{r1} and {r2}");
    // Variables r1 and r2 will not be used after this point.

    let r3 = &mut s; // no problem
    println!("{r3}");

}