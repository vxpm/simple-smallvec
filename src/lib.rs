use std::{
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

/// The inline data of the [`Smallvec`]. It's what's
/// stored on the stack while the length of the [`Smallvec`]
/// is `<= capacity`.
struct Inline<T, const CAPACITY: usize> {
    data: [MaybeUninit<T>; CAPACITY],
    len: usize,
}

impl<T, const CAPACITY: usize> Inline<T, CAPACITY> {
    const UNINIT: MaybeUninit<T> = MaybeUninit::uninit();

    /// Create a new, empty [`Inline`].
    #[inline]
    pub const fn new() -> Self {
        Inline {
            data: [Self::UNINIT; CAPACITY],
            len: 0,
        }
    }

    /// Returns the initialized slice of this [`Inline`].
    #[inline]
    pub fn initialized(&self) -> &[T] {
        // SAFETY: this is safe because:
        // 1. the pointer cast is fine because T and MaybeUninit<T>
        //    have the same layout
        // 2. all the invariants of 'as_ref' are upheld
        unsafe {
            (std::ptr::addr_of!(self.data[..self.len]) as *const [T])
                .as_ref()
                .unwrap()
        }
    }

    /// Returns the initialized, mutable, slice of this [`Inline`].
    #[inline]
    pub fn initialized_mut(&mut self) -> &mut [T] {
        // SAFETY: this is safe because:
        // 1. the pointer cast is fine because T and MaybeUninit<T>
        //    have the same layout
        // 2. all the invariants of 'as_mut' are upheld
        unsafe {
            (std::ptr::addr_of_mut!(self.data[..self.len]) as *mut [T])
                .as_mut()
                .unwrap()
        }
    }

    /// Tries to push the given value into the remaining space.
    ///
    /// # Errors
    /// If full, returns `Err(value)`.
    #[inline]
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == CAPACITY {
            return Err(value);
        }

        self.data[self.len].write(value);
        self.len += 1;

        Ok(())
    }

    /// Pops a value.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        (self.len > 0).then(|| {
            self.len -= 1;

            // SAFETY: this is safe because the last initialized
            // element will always be at index length - 1.
            // note that len is used because len was already
            // updated in the line above
            unsafe { self.data[self.len].assume_init_read() }
        })
    }
}

impl<T, const CAPACITY: usize> Drop for Inline<T, CAPACITY> {
    #[inline]
    fn drop(&mut self) {
        for elem in self.data[..self.len].iter_mut() {
            // SAFETY: this is safe because only values
            // within the initialized range are dropped
            unsafe {
                elem.assume_init_drop();
            }
        }
    }
}

impl<T, const CAPACITY: usize> Clone for Inline<T, CAPACITY>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut cloned_data = [Self::UNINIT; CAPACITY];
        for (src, dst) in self.initialized().iter().zip(cloned_data.iter_mut()) {
            dst.write(src.clone());
        }

        Self {
            data: cloned_data,
            len: self.len,
        }
    }
}

impl<T, const CAPACITY: usize> PartialEq for Inline<T, CAPACITY>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        let slice_a = self.initialized();
        let slice_b = other.initialized();

        slice_a.eq(slice_b)
    }
}

impl<T, const CAPACITY: usize> Eq for Inline<T, CAPACITY> where T: PartialEq + Eq {}

/// The inner data of a [`Smallvec`]. This could very well be
/// the [`Smallvec`] type, but that would expose the enum variants
/// and, therefore, it's contents - which is a no no.
#[derive(Clone, PartialEq, Eq)]
enum Inner<T, const CAPACITY: usize> {
    Stack(Inline<T, CAPACITY>),
    Heap(Vec<T>),
}

/// A Smallvec. Works just like a normal [`Vec<T>`], but it stores
/// it's elements in the stack up until a certain threshold. When this
/// threshold gets surpassed, the data is moved to the heap (i.e.
/// an actual [`Vec<T>`]).
///
/// `CAPACITY` is the amount of elements a [`Smallvec`] can hold on
/// the stack. It's given in elements (not bytes).
///
/// # Creating a new Smallvec
/// The simplest (but not only) way is to use the [`smallvec`] macro:
/// ```
/// # #[macro_use] extern crate simple_smallvec;
/// # use simple_smallvec::Smallvec;
/// # fn main() {
/// // creates an empty smallvec with inferred capacity and type.
/// let v: Smallvec<String, 16> = smallvec![];
///
/// // creates an empty smallvec with capacity of 16 and inferred type.
/// let mut v = smallvec![16;];
/// v.push(String::from("inferred to be String!"));
///
/// // creates an smallvec with capacity of 16 and the given elements.
/// // type is inferred from elements
/// let v = smallvec![16; String::from("hello"), String::from("world")];
///
/// // creates an smallvec with the given elements and capacity equal
/// // to the amount of elements. type is inferred from elements
/// let v = smallvec![String::from("hello"), String::from("world")];
/// # }
/// ```
#[derive(Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct Smallvec<T, const CAPACITY: usize>(Inner<T, CAPACITY>);

impl<T, const CAPACITY: usize> Smallvec<T, CAPACITY> {
    /// Creates a new [`Smallvec`].
    #[inline]
    pub const fn new() -> Self {
        Self(Inner::Stack(Inline::new()))
    }

    /// Creates a new [`Smallvec`] with the given capacity. That is,
    /// if `capacity < Self::CAPACITY` (the given capacity is less than
    /// the inline capacity) then contents of the [`Smallvec`] will be
    /// put on the heap directly.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity > CAPACITY {
            Self(Inner::Heap(Vec::with_capacity(capacity)))
        } else {
            Self(Inner::Stack(Inline::new()))
        }
    }

    /// Try to turn this [`Smallvec`] into it's inner [`Vec<T>`].
    ///
    /// # Errors
    /// If the contents are being stored inline, returns `Err(Self)`.
    #[inline]
    pub fn into_inner_vec(self) -> Result<Vec<T>, Self> {
        match self.0 {
            Inner::Stack(_) => Err(self),
            Inner::Heap(v) => Ok(v),
        }
    }

    /// Whether this [`Smallvec`] is storing it's contents inline or not.
    #[inline]
    pub const fn inline(&self) -> bool {
        match self.0 {
            Inner::Stack(_) => true,
            Inner::Heap(_) => false,
        }
    }

    /// Push `value` to the end of this [`Smallvec`].
    #[inline]
    pub fn push(&mut self, value: T) {
        match &mut self.0 {
            Inner::Stack(inline) => {
                if let Err(value) = inline.try_push(value) {
                    // move data on stack to heap
                    let Inline { data, len } = inline;
                    let mut vec: Vec<T> = Vec::with_capacity(CAPACITY + 1);

                    // copy the array to the vec
                    // SAFETY: this is safe because the array is fully
                    // initialized (since len == capacity) and the vec
                    // has enough room for all its elements (plus one!)
                    unsafe {
                        let ptr = vec.as_mut_ptr();

                        // this ptr cast is safe because MaybeUninit<T>
                        // has the same layout as T
                        let src = data.as_ptr() as _;
                        ptr.copy_from_nonoverlapping(src, CAPACITY);

                        // now, copy the remaining new element
                        // this unwrap is safe because the vec has a capacity
                        // of at least 1 (since it's CAPACITY + 1)
                        vec.spare_capacity_mut().last_mut().unwrap().write(value);

                        // finally, set the length of the vector
                        vec.set_len(*len + 1);
                    }

                    // here we forget the inline inner so that it's
                    // drop impl doesn't run and therefore the drop impl
                    // of the contents does not too
                    let inline_inner = std::mem::replace(&mut self.0, Inner::Heap(vec));
                    std::mem::forget(inline_inner);
                }
            }
            Inner::Heap(v) => v.push(value),
        }
    }

    /// Pops a value from this [`Smallvec`].
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.0 {
            Inner::Stack(inline) => inline.pop(),
            Inner::Heap(v) => v.pop(),
        }
    }
}

impl<T, const CAPACITY: usize> Deref for Smallvec<T, CAPACITY> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        match &self.0 {
            Inner::Stack(inline) => inline.initialized(),
            Inner::Heap(v) => v.deref(),
        }
    }
}

impl<T, const CAPACITY: usize> DerefMut for Smallvec<T, CAPACITY> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.0 {
            Inner::Stack(inline) => inline.initialized_mut(),
            Inner::Heap(v) => v.deref_mut(),
        }
    }
}

impl<T, const CAPACITY: usize> std::fmt::Debug for Smallvec<T, CAPACITY>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const CAPACITY: usize> FromIterator<T> for Smallvec<T, CAPACITY> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower_bound, upper_bound) = iter.size_hint();

        let mut v = Smallvec::with_capacity(upper_bound.unwrap_or(lower_bound));
        for element in iter {
            v.push(element);
        }

        v
    }
}

/// See [`Smallvec`] for usage.
#[macro_export]
macro_rules! smallvec {
    (@count_exprs $first:expr, $($rest:expr),*) => {
        1 + smallvec!(@count_exprs $($rest),*)
    };
    (@count_exprs $value:expr) => { 1 };
    ($len:expr; $($elem:expr),* $(,)?) => {
        {
            let mut v: Smallvec<_, { $len }> = Smallvec::new();
            $(
                v.push($elem);
            )*

            v
        }
    };
    ($($elem:expr),+ $(,)?) => {
        {
            const COUNT: usize = smallvec!(@count_exprs $($elem),*);
            smallvec![COUNT; $($elem),*]
        }
    };
    () => {
        Smallvec::new()
    };
}

#[cfg(test)]
mod tests {
    use super::Smallvec;

    #[test]
    fn basic_usage() {
        let mut v = smallvec![
            String::from("hello"),
            String::from("world"),
            String::from("test"),
        ];

        assert_eq!(v[0], "hello");
        assert_eq!(v[1], "world");
        assert_eq!(v[2], "test");
        assert!(v.inline());

        v.push(String::from("foo!"));

        assert_eq!(v[0], "hello");
        assert_eq!(v[1], "world");
        assert_eq!(v[2], "test");
        assert_eq!(v[3], "foo!");
        assert!(!v.inline());

        assert_eq!(v.pop().as_deref(), Some("foo!"));
        assert_eq!(v.pop().as_deref(), Some("test"));
        assert!(!v.inline());

        assert_eq!(v.pop().as_deref(), Some("world"));
        assert_eq!(v.pop().as_deref(), Some("hello"));
        assert_eq!(v.pop().as_deref(), None);
        assert!(!v.inline());
    }

    #[test]
    fn to_inner_vec() {
        let v = smallvec![
            String::from("hello"),
            String::from("world"),
            String::from("test")
        ];

        let mut v = v.into_inner_vec().expect_err("should not be on the heap");

        v.push(String::from("foo!"));

        v.into_inner_vec().expect("should be on the heap");
    }

    #[test]
    fn clone() {
        let v_0 = smallvec![
            String::from("hello"),
            String::from("world"),
            String::from("test")
        ];

        let mut v_1 = v_0.clone();

        v_1.push(String::from("foo!"));

        assert_eq!(v_0[0], "hello");
        assert_eq!(v_0[1], "world");
        assert_eq!(v_0[2], "test");
        assert!(v_0.inline());

        assert_eq!(v_1[0], "hello");
        assert_eq!(v_1[1], "world");
        assert_eq!(v_1[2], "test");
        assert_eq!(v_1[3], "foo!");
        assert!(!v_1.inline());
    }

    #[test]
    fn eq() {
        let v_0 = smallvec![
            String::from("hello"),
            String::from("world"),
            String::from("test")
        ];

        let v_1 = v_0.clone();

        assert_eq!(v_0, v_1);
    }
}
