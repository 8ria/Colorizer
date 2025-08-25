use actix_files::{Files, NamedFile};
use actix_governor::{Governor, GovernorConfigBuilder};
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use ort::{environment::Environment, session::Session, session::SessionBuilder, tensor::OrtOwnedTensor, value::Value};
use serde::{Deserialize, Serialize};
use std::{error::Error, fs::File, path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;

/// Input JSON: `{ "text": "example sentence" }`
#[derive(Deserialize)]
struct TextInput {
    text: String,
}

/// Output JSON: `{ "r": 123, "g": 45, "b": 67 }`
#[derive(Serialize)]
struct ColorOutput {
    r: u8,
    g: u8,
    b: u8,
}

/// Reference embedding with an associated RGB color.
#[derive(Deserialize, Serialize)]
struct RefEmbedding {
    embedding: Vec<f32>,
    color: (u8, u8, u8),
}

/// Shared application state
struct AppState {
    tokenizer: Tokenizer,
    session: Session,
    ref_embeddings: Vec<RefEmbedding>,
}

/// Compute cosine similarity between two embeddings
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Generate an embedding for a sentence using the tokenizer + ONNX model
fn get_embedding(
    tokenizer: &Tokenizer,
    session: &Session,
    sentence: &str,
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    let encoding = tokenizer.encode(sentence, true)?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

    let seq_len = input_ids.len();
    let input_ids_arr = ndarray::Array2::from_shape_vec((1, seq_len), input_ids)?.into_dyn();
    let attention_mask_arr = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask)?.into_dyn();

    let input_ids_cow = ndarray::CowArray::from(input_ids_arr);
    let attention_mask_cow = ndarray::CowArray::from(attention_mask_arr);

    let input_ids_val = Value::from_array(session.allocator(), &input_ids_cow)?;
    let attention_mask_val = Value::from_array(session.allocator(), &attention_mask_cow)?;

    let outputs = session.run(vec![input_ids_val, attention_mask_val])?;
    let tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let arr = tensor.view();

    // Pooling by averaging token embeddings
    let summed = arr.index_axis(ndarray::Axis(0), 0).sum_axis(ndarray::Axis(0));
    let pooled = summed.clone() / summed.len() as f32;

    Ok(pooled.into_raw_vec())
}

/// POST /color → returns the closest color for input text
#[post("/color")]
async fn color(data: web::Data<AppState>, input: web::Json<TextInput>) -> impl Responder {
    match get_embedding(&data.tokenizer, &data.session, &input.text) {
        Ok(sentence_emb) => {
            let (mut best_color, mut best_sim) = ((0, 0, 0), f32::MIN);

            for ref_emb in &data.ref_embeddings {
                let sim = cosine_similarity(&sentence_emb, &ref_emb.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_color = ref_emb.color;
                }
            }

            HttpResponse::Ok().json(ColorOutput {
                r: best_color.0,
                g: best_color.1,
                b: best_color.2,
            })
        }
        Err(e) => HttpResponse::InternalServerError().body(e.to_string()),
    }
}

/// GET / → serves `static/index.html` if available
#[get("/")]
async fn index(req: HttpRequest) -> actix_web::Result<impl Responder> {
    let path: PathBuf = "./static/index.html".into();
    if path.exists() {
        Ok(NamedFile::open(path)?.into_response(&req))
    } else {
        Ok(HttpResponse::NotFound().finish())
    }
}

/// Application entrypoint
#[actix_web::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("🚀 Server starting at: http://localhost:8090/");

    // Load tokenizer + ONNX model
    let tokenizer = Tokenizer::from_file("models/tokenizer.json")?;
    let environment = Arc::new(Environment::builder().with_name("default").build()?);
    let session = SessionBuilder::new(&environment)?.with_model_from_file("models/model.onnx")?;

    // Load reference embeddings
    let file = File::open("custom/ref_embeddings.json")?;
    let ref_embeddings: Vec<RefEmbedding> = serde_json::from_reader(file)?;

    // Shared app state
    let state = web::Data::new(AppState {
        tokenizer,
        session,
        ref_embeddings,
    });

    // Rate limiting
    let governor_conf = GovernorConfigBuilder::default()
        .milliseconds_per_request(200)
        .burst_size(10)
        .finish()
        .unwrap();

    // Launch server
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(Governor::new(&governor_conf))
            .service(Files::new("/static", "./static").show_files_listing())
            .service(index)
            .service(color)
    })
    .bind(("0.0.0.0", 8090))?
    .run()
    .await?;

    Ok(())
}
