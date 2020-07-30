const { Schema, model } = require('mongoose');

const imageSchema = new Schema({
    ns: { type: String, default: "imgpdf.png" },
    predic: { type: String },
    nuevo: { type: String, default: null },
    text: { type: String },
    title: { type: String },
    description: { type: String },
    filename: { type: String },
    path: { type: String },
    originalname: { type: String },
    mimetype: { type: String },
    size: { type: Number },
    user: { type: String, required: true },
    created_at: { type: Date, default: Date.now() }
});

module.exports = model('Image', imageSchema);