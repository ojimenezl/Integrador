const { Schema, model } = require('mongoose');

const docSchema = new Schema({
    res: { type: String },
    hecho: { type: String },
    descripcion: { type: String },
    pruebas: { type: String },
    rsi: { type: String },
    rno: { type: String },
    created_at: { type: Date, default: Date.now() }
});

module.exports = model('Documento', docSchema);