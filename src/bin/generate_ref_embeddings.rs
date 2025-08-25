use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis, CowArray};
use ort::{
    environment::Environment,
    session::{Session, SessionBuilder},
    tensor::OrtOwnedTensor,
    value::Value,
};
use serde::Serialize;
use serde_json;
use std::{fs::File, sync::Arc};
use tokenizers::Tokenizer;

/// A reference embedding tied to a color.
#[derive(Serialize)]
struct RefEmbedding {
    embedding: Vec<f32>,
    color: (u8, u8, u8),
}

/// Generate an embedding for a sentence using the tokenizer + ONNX model.
fn get_embedding(tokenizer: &Tokenizer, session: &Session, sentence: &str) -> Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(sentence, true)
        .map_err(|e| anyhow!("Tokenizer error: {}", e))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

    let seq_len = input_ids.len();
    let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids)?.into_dyn();
    let attention_mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask)?.into_dyn();

    // ✅ Bind temporaries so they live long enough
    let input_ids_cow = CowArray::from(input_ids_arr);
    let attention_mask_cow = CowArray::from(attention_mask_arr);

    let input_ids_val = Value::from_array(session.allocator(), &input_ids_cow)?;
    let attention_mask_val = Value::from_array(session.allocator(), &attention_mask_cow)?;

    let outputs = session.run(vec![input_ids_val, attention_mask_val])?;
    let tensor: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let arr = tensor.view();

    // Average token embeddings
    let summed = arr.index_axis(Axis(0), 0).sum_axis(Axis(0));
    let pooled = summed.clone() / summed.len() as f32;

    Ok(pooled.into_raw_vec())
}

/// Entrypoint: generates `custom/ref_embeddings.json`
fn main() -> Result<()> {
    println!("📦 Generating reference embeddings...");

    // Load tokenizer + ONNX model
    let tokenizer = Tokenizer::from_file("models/tokenizer.json")
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    let environment = Arc::new(Environment::builder().with_name("default").build()?);
    let session = SessionBuilder::new(&environment)?.with_model_from_file("models/model.onnx")?;

    // Words mapped to representative RGB colors
    let ref_words = vec![
        ("love", (255, 0, 0)),
        ("sun", (255, 255, 0)),
        ("girl", (255, 110, 240)),
        ("boy", (0, 0, 255)),
        ("happy", (255, 200, 0)),
        ("sad", (0, 0, 255)),
        ("anger", (200, 0, 0)),
        ("calm", (0, 255, 200)),
        ("fear", (50, 0, 50)),
        ("joy", (255, 150, 0)),
        ("peace", (0, 255, 100)),
        ("trust", (0, 120, 255)),
        ("hate", (150, 0, 0)),
        ("fun", (255, 180, 0)),
        ("lonely", (100, 100, 200)),
        ("excited", (255, 90, 0)),
        ("bored", (150, 150, 150)),
        ("cute", (255, 160, 220)),
        ("dream", (180, 0, 255)),
        ("music", (0, 200, 255)),
        ("rain", (0, 100, 200)),
        ("flower", (255, 180, 180)),
        ("nature", (0, 200, 100)),
        ("child", (255, 220, 180)),
        ("loveable", (255, 0, 100)),
        ("warm", (255, 120, 0)),
        ("cold", (0, 180, 255)),
        ("smile", (255, 220, 0)),
        ("tears", (0, 50, 255)),
        ("adventure", (255, 140, 0)),
        ("hope", (0, 255, 150)),
        ("dreamy", (200, 100, 255)),
        ("mystery", (50, 0, 100)),
        ("energy", (255, 80, 0)),
        ("fearless", (255, 0, 50)),
        ("calmness", (0, 255, 200)),
        ("magic", (200, 0, 255)),
        ("freedom", (0, 255, 100)),
        ("curious", (255, 200, 100)),
        ("bold", (255, 0, 0)),
        ("gentle", (100, 200, 255)),
        ("soft", (200, 255, 255)),
        ("storm", (100, 0, 100)),
        ("warmth", (255, 100, 0)),
        ("loneliness", (80, 80, 200)),
        ("passion", (255, 0, 0)),
        ("confidence", (255, 150, 0)),
        ("angerous", (180, 0, 0)),
        ("joyful", (255, 200, 0)),
        ("sorrow", (0, 0, 150)),
        ("care", (0, 200, 150)),
        ("gloom", (50, 50, 100)),
        ("excitement", (255, 90, 0)),
        ("lovebird", (255, 0, 100)),
        ("peaceful", (0, 255, 150)),
        ("curiosity", (255, 180, 100)),
        ("playful", (255, 150, 200)),
        ("romance", (255, 0, 50)),
        ("affection", (255, 50, 100)),
        ("delight", (255, 200, 50)),
        ("comfort", (100, 255, 200)),
        ("melancholy", (0, 50, 200)),
        ("optimism", (255, 220, 0)),
        ("pessimism", (50, 0, 100)),
        ("trustworthy", (0, 120, 255)),
        ("friendship", (100, 200, 255)),
        ("admire", (255, 180, 100)),
        ("surprise", (255, 220, 50)),
        ("shock", (200, 0, 0)),
        ("confident", (255, 140, 0)),
        ("timid", (100, 150, 255)),
        ("energetic", (255, 80, 0)),
        ("lively", (255, 180, 0)),
        ("sadness", (0, 0, 200)),
        ("grief", (50, 0, 150)),
        ("hopeful", (0, 255, 100)),
        ("relax", (0, 200, 200)),
        ("bliss", (255, 220, 100)),
        ("cheerful", (255, 180, 0)),
        ("envy", (0, 150, 0)),
        ("jealousy", (50, 100, 0)),
        ("fearful", (50, 0, 50)),
        ("brave", (255, 50, 0)),
        ("nervous", (200, 100, 50)),
        ("peacekeeper", (0, 255, 150)),
        ("lucky", (255, 255, 0)),
        ("unlucky", (50, 50, 50)),
        ("grateful", (255, 200, 50)),
        ("thankful", (255, 180, 50)),
        ("romantic", (255, 0, 100)),
        ("friendly", (0, 200, 255)),
        ("thoughtful", (150, 200, 255)),
        ("hopefulness", (0, 255, 120)),
        ("curiousness", (255, 180, 100)),
        ("joyfulness", (255, 200, 0)),
        ("sadistic", (100, 0, 0)),
        ("lonelyness", (80, 80, 200)),
        ("romanticize", (255, 0, 50)),
        ("comforting", (100, 255, 200)),
        ("delighted", (255, 200, 50)),
        ("excitedly", (255, 90, 0)),
        ("energetically", (255, 80, 0)),
        ("blissful", (255, 220, 100)),
        ("calmly", (0, 255, 200)),
        ("gloomy", (50, 50, 100)),
        ("romanceful", (255, 0, 50)),
        ("cheery", (255, 180, 0)),
        ("mysterious", (50, 0, 100)),
        ("playfully", (255, 150, 200)),
        ("fearlessly", (255, 0, 50)),
        ("sadfully", (0, 0, 200)),
        ("thoughtfully", (150, 200, 255)),
        ("friendlily", (0, 200, 255)),
        ("trustfully", (0, 120, 255)),
        ("optimistically", (255, 220, 0)),
        ("pessimistically", (50, 0, 100)),
        ("gratefully", (255, 200, 50)),
        ("thankfully", (255, 180, 50)),
        ("cheerfully", (255, 180, 0)),
        ("romantically", (255, 0, 100)),
        ("peacefully", (0, 255, 150)),
        ("nervously", (200, 100, 50)),
        ("adventurous", (255, 140, 0)),
        ("curiously", (255, 180, 100)),
        ("magical", (200, 0, 255)),
        ("dreamily", (180, 0, 255)),
        ("lovely", (255, 0, 0)),
        ("warmhearted", (255, 120, 0)),
        ("coldhearted", (0, 180, 255)),
        ("exciting", (255, 90, 0)),
        ("peaceableness", (0, 255, 100)),
        ("playfulness", (255, 150, 200)),
        ("friendliness", (0, 200, 255)),
        ("happiness", (255, 200, 0)),
        ("sadnessful", (0, 0, 255)),
        ("fearfulness", (50, 0, 50)),
        ("angerful", (200, 0, 0)),
        ("calmful", (0, 255, 200)),
        ("trustful", (0, 120, 255)),
        ("joyous", (255, 150, 0)),
        ("sorrowful", (0, 0, 150)),
        ("energetical", (255, 80, 0)),
        ("blissfulness", (255, 220, 100)),
        ("cheerfulness", (255, 180, 0)),
        ("curiousful", (255, 180, 100)),
        ("romancical", (255, 0, 50)),
        ("friendfully", (0, 200, 255)),
        ("magically", (200, 0, 255)),
        ("hopefully", (0, 255, 100)),
        ("delightful", (255, 200, 50)),
        ("adventurously", (255, 140, 0)),
        ("mysteriously", (50, 0, 100)),
        ("calmnessful", (0, 255, 200)),
        ("excitedful", (255, 90, 0)),
        ("playfulnessful", (255, 150, 200)),
        ("fearlesslyful", (255, 0, 50)),
        ("lovefully", (255, 0, 0)),
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("yellow", (255, 255, 0)),
        ("cyan", (0, 255, 255)),
        ("magenta", (255, 0, 255)),
        ("orange", (255, 165, 0)),
        ("purple", (128, 0, 128)),
        ("pink", (255, 192, 203)),
        ("brown", (165, 42, 42)),
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("gray", (128, 128, 128)),
        ("lime", (0, 255, 0)),
        ("navy", (0, 0, 128)),
        ("teal", (0, 128, 128)),
        ("olive", (128, 128, 0)),
        ("maroon", (128, 0, 0)),
        ("silver", (192, 192, 192)),
        ("gold", (255, 215, 0)),
        ("violet", (238, 130, 238)),
        ("indigo", (75, 0, 130)),
        ("turquoise", (64, 224, 208)),
        ("beige", (245, 245, 220)),
        ("coral", (255, 127, 80)),
        ("salmon", (250, 128, 114)),
        ("khaki", (240, 230, 140)),
        ("lavender", (230, 230, 250)),
        ("peach", (255, 218, 185)),
        ("mint", (189, 252, 201)),
        ("apricot", (251, 206, 177)),
        ("crimson", (220, 20, 60)),
        ("azure", (0, 127, 255)),
        ("emerald", (80, 200, 120)),
        ("ruby", (224, 17, 95)),
        ("sapphire", (15, 82, 186)),
        ("amethyst", (153, 102, 204)),
        ("carmine", (150, 0, 24)),
        ("cerulean", (42, 82, 190)),
        ("periwinkle", (204, 204, 255)),
        ("chartreuse", (127, 255, 0)),
        ("tan", (210, 180, 140)),
        ("indianred", (205, 92, 92)),
        ("orchid", (218, 112, 214)),
        ("plum", (221, 160, 221)),
        ("seafoam", (159, 226, 191)),
        ("mustard", (255, 219, 88)),
        ("blush", (222, 93, 131)),
        ("shit", (117, 99, 0)),
        ("sky", (130, 228, 255)),
        ("cloud", (130, 228, 255)),
        ("snow", (247, 247, 247)),
        ("ocean", (6, 66, 115)),
        ("forest", (1, 68, 33)),
        ("tree", (12, 174, 91)),
        ("flower", (249, 213, 229)),
        ("desert", (193, 154, 107)),
        ("sand", (203, 189, 147)),
        ("stone", (227, 203, 165)),
        ("fire", (255, 0, 0)),
        ("ice", (63, 208, 212)),
        ("sunset", (238, 93, 108)),
        ("sunrise", (253, 236, 167)),
        ("banana", (251, 236, 93)),
        ("tomato", (220, 20, 60)),
        ("lemon", (255,	247, 0)),
        ("cherry", (227, 2, 2)),
        ("fucking", (255, 0, 0)),
        ("carrot", (237, 145, 33)),
        ("pumpkin", (255, 117, 24)),
        ("chocolate", (113, 54, 0)),
        ("gold", (207, 181, 59)),
        ("silver", (192, 192, 192)),
        ("diamond", (241, 247, 251)),
        ("blood", (116, 7, 7)),
        ("smoke", (216, 216, 216)),
        ("energy", (255, 200, 0)),
        ("power", (200, 0, 0)),
        ("freedom", (64, 224, 208)),
        ("danger", (255, 69, 0)),
        ("luck", (0, 200, 0)),
        ("time", (70, 130, 180)),
        ("knowledge", (0, 102, 204)),
        ("death", (48, 0, 48)),
        ("life", (124, 252, 0)),
        ("growth", (50, 205, 50)),
        ("decay", (128, 128, 0)),
    ];

    // Build embeddings
    let mut ref_embeddings = Vec::new();
    for (word, rgb) in ref_words {
        let emb = get_embedding(&tokenizer, &session, word)?;
        ref_embeddings.push(RefEmbedding {
            embedding: emb,
            color: rgb,
        });
        println!("  ✓ Embedded word: {}", word);
    }

    // Save to JSON
    let file = File::create("custom/ref_embeddings.json")?;
    serde_json::to_writer_pretty(file, &ref_embeddings)?;
    println!("✅ Saved reference embeddings → custom/ref_embeddings.json");

    Ok(())
}
