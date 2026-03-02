pub mod corvia {
    pub mod inference {
        pub mod v1 {
            tonic::include_proto!("corvia.inference.v1");
        }
    }
}

// Convenience re-export
pub use corvia::inference::v1::*;
