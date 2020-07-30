const { Router } = require('express');
const path = require('path');
const { unlink } = require('fs-extra');
// Helpers
const { isAuthenticated } = require("../helpers/auth");
const router = Router();
// Models
const Image = require('../models/Image');
const Imagetxt = require('../models/documento');

router.get('/repositorio', isAuthenticated, async(req, res) => {
    const images = await Image.find({ user: req.user.id }).sort({ date: "desc" }).lean();

    res.render('repositorio', { images });
});

router.get('/resumenes', isAuthenticated, async(req, res) => {
    const images = await Image.find({ user: req.user.id }).sort({ date: "desc" }).lean();

    res.render('resumenes', { images });
});

router.get('/redaccion', isAuthenticated, async(req, res) => {
    const images = await Image.find({ user: req.user.id }).sort({ date: "desc" }).lean();

    res.render('redaccion', { images });
});

router.get('/redaccion/pred', isAuthenticated, async(req, res) => {

    console.log(par)
    res.render('redaccion', { par });


});



router.get('/upload', isAuthenticated, async(req, res) => {
    res.render('upload');
});

router.get('/uploadimg', isAuthenticated, async(req, res) => {
    res.render('uploadimg');
});

router.post('/upload', isAuthenticated, async(req, res) => {
    const image = new Image();
    image.path = '/img/uploads/' + req.file.filename;
    const pyth = require('../ocrnode');
    arg1 = image.path
    console.log("aqui1", arg1)
    pyth.publicar(arg1, req, res).then(respu => publicar(respu, req, res)).then(resf => { console.log(resf) })



});


router.post('/uploadimg', isAuthenticated, async(req, res) => {
    const image = new Image();
    image.path = '/img/uploads/' + req.file.filename;
    const pyth = require('../imgnode');
    arg1 = image.path
    pyth.publicar(arg1, req, res).then(respu => publicarimg(respu, req, res)).then(resf => { console.log(resf) })



});

router.post('/redacion/prediccion', isAuthenticated, async(req, res) => {
    //const documento = new Imagetxt();
    const pyth = require('../predline');
    hecho = req.body.hecho;
    peticion = req.body.peticion;
    pruebas = req.body.pruebas;
    rsi = req.body.rsi;

    if (rsi === "on") {
        rsi = "padre biolígico"
        console.log(rsi)
    } else {
        rsi = "presunción de paternidad"
        console.log(rsi)
    }
    rno = req.body.rno;
    arg0 = hecho
    arg2 = peticion
    arg3 = pruebas
    arg4 = rsi

    arg1 = arg0 + " " + arg2 + " " + arg3 + " " + arg4
    console.log("aqui1", arg1)
    pyth.publicar(arg1, req, res).then(respu => publicaronline(hecho, peticion, pruebas, respu, req, res)).then(resf => { console.log(resf) })



});

async function publicaronline(hecho, peticion, pruebas, respu, req, res) {

    //const documentos = new Imagetxt();

    resm = respu.toString('utf8');

    const o = JSON.parse(resm);
    console.log(o.resm)
    let pares = {
        H: hecho,
        P: peticion,
        PB: pruebas,
        NB: o.NB,
        SVM: o.SVM,
        RL: o.RL,
        KNN: o.KNN,
        predi: o.res
    }

    console.log(pares)
    req.flash("success_msg", "Predicción Lista!!");
    res.render('redaccion', { pares });



}

async function publicar(respu, req, res) {

    const image = new Image();

    image.predic = respu.toString('utf8');
    image.title = req.body.title;
    image.description = req.body.description;
    image.filename = req.file.filename;
    image.path = '/img/uploads/' + req.file.filename;
    image.originalname = req.file.originalname;
    image.mimetype = req.file.mimetype;
    image.size = req.file.size;
    image.user = req.user.id;
    req.flash("success_msg", "Documento Guardado!!");
    await image.save();

    res.redirect('/repositorio');


}
async function publicarimg(respu, req, res) {

    const image = new Image();

    image.text = respu.toString('utf8');
    image.title = req.body.title;
    image.description = req.body.description;
    image.filename = req.file.filename;
    image.path = '/img/uploads/' + req.file.filename;
    image.originalname = req.file.originalname;
    image.mimetype = req.file.mimetype;
    image.size = req.file.size;
    image.user = req.user.id;
    req.flash("success_msg", "Documento Guardado!!");

    await image.save();

    res.redirect('/repositorio');


}
router.put("/image/:id", async(req, res) => {
    const { nombrecedula, celular, correo, nombrecedula2, DEM } = req.body;
    console.log(celular)
    console.log(DEM, "DEMMMMMMMM")
    var nuevo = "{ \"nombrecedula\": \"" + nombrecedula + "\", \"celular\":\"" + celular + "\", \"correo\":\"" + correo + "\",\"nombrecedula2\":\"" + nombrecedula2 + "\",\"DEM\":\"" + DEM + "\"}"
    await Image.findByIdAndUpdate(req.params.id, {
        nuevo
    });
    req.flash("success_msg", "Cambio Realizado");
    res.redirect("/repositorio");
});

router.post('/image/:id', async(req, res, next) => {
    const { id } = req.params;
    await Image.update({ _id: id }, req.body);
    res.redirect('/image/:id');
});


router.get('/image/:id', isAuthenticated, async(req, res) => {
    const { id } = req.params;
    const images = await Image.findById(id)
    const idd = images.id

    if (!images.predic) {
        var s2 = images.text
        var path = images.path
        var filename = images.filename
        var title = images.title
        var description = images.description
        console.log(s2)

        const cm = JSON.parse(s2);

        let docu = { text: cm.text, path: images.path, filename: images.filename, title: images.title, description: images.description }
        console.log(docu)
        res.render('profileimg', { docu });
    } else {
        var s = images.predic
        console.log(s)
        var o = JSON.parse(s);

        if (!images.nuevo) {
            console.log(o.celular)
            var nom1 = o.nombrecedula
            var cel = o.celular
            var correo = o.correo
            var nom2 = o.nombrecedula2
            var DEM = o.DEM

        } else {
            var s = images.nuevo
            console.log("aquiii-----")
            console.log(s)
            var op = JSON.parse(s);
            console.log(op)
            var nom1 = op.nombrecedula

            var cel = op.celular

            var correo = op.correo

            var nom2 = op.nombrecedula2

            var DEM = op.DEM
            console.log(DEM)
        }

        let par = {
            id: idd,
            nom1: nom1,
            cel: cel,
            correo: correo,
            nom2: nom2,
            NB: o.NB,
            ANB: o.ANB,
            PNB: o.PNB,
            RNB: o.RNB,
            FNB: o.FNB,
            SVM: o.SVM,
            ASVM: o.ASVM,
            PSVM: o.PSVM,
            RSVM: o.RSVM,
            FSVM: o.FSVM,
            RL: o.RL,
            ARL: o.ARL,
            PRL: o.PRL,
            RRL: o.RRL,
            FRL: o.FRL,
            KNN: o.KNN,
            AKNN: o.AKNN,
            PKNN: o.PKNN,
            RKNN: o.RKNN,
            FKNN: o.FKNN,
            DEM: DEM,
            predi: o.res
        }
        console.log(par)
        res.render('profile', { par });



    }


});

router.delete('/imagedelete/:id', isAuthenticated, async(req, res) => {



    await Image.findByIdAndDelete(req.params.id);
    req.flash("success_msg", "ELIMINADO");
    const { id } = req.params;
    const imageDeleted = await Image.findByIdAndDelete(id);
    res.redirect('/repositorio');


});
router.delete('/imagedeletetab/:id', isAuthenticated, async(req, res) => {



    await Image.findByIdAndDelete(req.params.id);
    req.flash("success_msg", "ELIMINADO");
    const { id } = req.params;
    const imageDeleted = await Image.findByIdAndDelete(id);
    res.redirect('/resumenes');


});

module.exports = router;