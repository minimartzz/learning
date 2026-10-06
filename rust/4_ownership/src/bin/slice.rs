// ==== Slice ========================================
// - Slice references a contiguous sequence of elements in a collection
// - A type of reference so it has no ownership

fn main() {
    let x = String::from("something here");

    let out = first_word(&x);

    println!("{out}");

    // Slices
    let something = &x[..9];
    let here = &x[10..14];
    println!("{something} {here}");

    // Comparing between string literals and string pointers
    another_fn();
}

fn first_word(s: &str) -> &str {
    let bytes = s.as_bytes();

    for (i, &item) in bytes.iter().enumerate() {
        if item == b' ' {
            return &s[..i];
        }
    }

    &s[..]
}

fn another_fn() {
    let my_string = String::from("hello world");

    // `first_word` works on slices of Strings whether partial or whole
    let word = first_word(&my_string[0..6]);
    let word = first_word(&my_string[..]);
    println!("{word}");

    // `first_word` also works on references to `String's` which are equivalent to whole slices of strings
    let word = first_word(&my_string);
    println!("{word}");

    let my_string_literal = "hello world";
    // `first_word` works on slices of string literals, whether partial or
    // whole.
    let word = first_word(&my_string_literal[0..6]);
    let word = first_word(&my_string_literal[..]);
    println!("{word}");

    // Because string literals *are* string slices already,
    // this works too, without the slice syntax!
    let word = first_word(my_string_literal);
    println!("{word}");
}