// nufrost-core — shared types, math, and algorithm foundations.
// Do NOT port NUFROST/HANTS/Zhu2015 reconstruction logic here yet.
// This is a placeholder skeleton.

/// Adds two integers together.
pub fn add(left: i32, right: i32) -> i32 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_add() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn placeholder_add_negative() {
        assert_eq!(add(-1, 1), 0);
    }
}
