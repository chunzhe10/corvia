fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &[
                "proto/corvia/inference/v1/embedding.proto",
                "proto/corvia/inference/v1/chat.proto",
                "proto/corvia/inference/v1/model.proto",
            ],
            &["proto/"],
        )?;
    Ok(())
}
