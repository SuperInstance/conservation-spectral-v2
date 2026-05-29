//! 64-byte aligned vector types for SIMD-friendly memory layout.
//!
//! Assembly insight: AVX-512 demands 64-byte alignment for aligned loads.
//! Every matrix and vector in this SDK is 64-byte aligned.

use std::alloc::{alloc, dealloc, Layout};

/// Required alignment for AVX-512 (64 bytes).
pub const ALIGN_64: usize = 64;

/// A `Vec<T>` with 64-byte aligned allocation, optimized for SIMD.
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> AlignedVec<T> {
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            cap: 0,
        }
    }

    /// Allocate `len` elements, zero-initialized (for f64/u64).
    pub fn zeroed(len: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        if len == 0 {
            return Self::new();
        }
        let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), ALIGN_64)
            .expect("layout overflow");
        // SAFETY: layout size > 0, alignment is valid
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // Zero the memory
        unsafe { std::ptr::write_bytes(ptr, 0u8, len * std::mem::size_of::<T>()) };
        Self {
            // SAFETY: ptr was allocated for T with correct alignment
            ptr: ptr as *mut T,
            len,
            cap: len,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr is valid for len elements
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: ptr is valid for len elements
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Convert from a regular Vec by re-allocating with 64-byte alignment.
    pub fn from_vec(v: Vec<T>) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let len = v.len();
        if len == 0 {
            std::mem::forget(v);
            return Self::new();
        }
        let mut aligned = Self::zeroed(len);
        // SAFETY: copy from vec to aligned buffer
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), aligned.ptr, len);
        }
        std::mem::forget(v); // don't drop the original
        aligned
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // Drop all elements
        unsafe {
            std::ptr::drop_in_place(std::slice::from_raw_parts_mut(self.ptr, self.len));
        }
        let layout = Layout::from_size_align(self.cap * std::mem::size_of::<T>(), ALIGN_64)
            .expect("layout overflow in drop");
        // SAFETY: ptr was allocated with this layout
        unsafe {
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

impl<T: Clone + bytemuck::Zeroable> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new = Self::zeroed(self.len);
        for i in 0..self.len {
            new[i] = self[i].clone();
        }
        new.len = self.len;
        new
    }
}

impl<T> std::ops::Index<usize> for AlignedVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: usize) -> &T {
        debug_assert!(index < self.len, "AlignedVec index out of bounds");
        // SAFETY: index checked above
        unsafe { &*self.ptr.add(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for AlignedVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len, "AlignedVec index out of bounds");
        // SAFETY: index checked above
        unsafe { &mut *self.ptr.add(index) }
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}
