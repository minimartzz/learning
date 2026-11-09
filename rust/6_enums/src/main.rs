// ==== Enums ========================================
// - Express one value will come from a set of values
enum IpAddrKind {
    V4,
    V6
}

fn main() {
    // Instances of enums
    // - Both are under the `IpAddrKind` type, so either can be used in function definitions
    let four = IpAddrKind::V4;
    let six = IpAddrKind::V6;

}
