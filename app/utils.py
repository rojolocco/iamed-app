
# Cargando librerias ##################################################
from app import app

import numpy as np
import pandas as pd
import base64
import sys
import os
from werkzeug.utils import secure_filename
#######################################################################


# Umbrales ############################################################
tresholds = {'Atelectasia': 0.5022405833005905,
                'Cardiomegalia': 0.18895688865865978,
                'Consolidación': 0.28814152224610246,
                'Edema': 0.5163687798711989,
                'Cardiomediastino Agrandado': 0.2852344489656389,
                'Fractura': 0.5,
                'Lesión Pulmonar': 0.5917971506714821,
                'Opacidad Pulmonar': 0.5416456088423729,
                'Sin Hallazgo': 0.5792196869850159,
                'Derrame Pleural': 0.46151544004678724,
                'Otro Pleural': 0.5022405833005905,
                'Neumonía': 0.4376093732813994,
                'Neumotórax': 0.3142763748764992,
                'Dispositivos de soporte': 0.5022405833005905}
#######################################################################


# Modelos escogidos ###################################################
pos_list = {'Atelectasis': np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            'Cardiomegaly': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0]),
            'Consolidation': np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
            'Edema': np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0]),
            'Enlarged Cardiomediastinum': np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]),
            'Fracture' : np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]),
            'Lung Lesion': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1]),
            'Lung Opacity': np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]),
            'No Finding': np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            'Pleural Effusion': np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]),
            'Pleural Other': np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0]),
            'Pneumonia': np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]),
            'Pneumothorax': np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            'Support Devices': np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0])}
#######################################################################


# Patologias ##########################################################
classes = ['Sin Hallazgo', 'Cardiomediastino Agrandado', 'Cardiomegalia',
            'Opacidad Pulmonar', 'Lesión Pulmonar', 'Edema', 'Consolidación', 'Neumonía',
            'Atelectasia', 'Neumotórax', 'Derrame Pleural', 'Otro Pleural',
            'Fractura', 'Dispositivos de soporte']

all_columns = {'0': 'No Finding', '1': 'Enlarged Cardiomediastinum', '2': 'Cardiomegaly', '3': 'Lung Opacity', '4': 'Lung Lesion',
                '5': 'Edema', '6': 'Consolidation', '7': 'Pneumonia', '8': 'Atelectasis', '9': 'Pneumothorax', '10': 'Pleural Effusion',
                '11': 'Pleural Other', '12': 'Fracture', '13': 'Support Devices'}
#######################################################################


# Lista de Patologias ##########################################
def patologies_preds(classes, tresholds, outputs):
    pat_list = []
    for idx, val in enumerate(classes):
        if outputs[idx] >= tresholds[val]:
            pat_list.append(val)
    return pat_list
#######################################################################


# Buscar Imagenes Permitidas ##########################################
def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOW_IMAGES_EXTENSIONS"]:
        return True
    else:
        return False
#######################################################################


# Codificaciones de Redis #############################################
def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):

    # if this is Python 3, we need the extra step of encoding the
    # serialized NumPy string as a byte object
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")

    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)

    # return the decoded image
    return a
#######################################################################


# Procesamiento de Predicciones #######################################
def pred_mean(y_predU_list):

        X_list = {}
        for k in range(14):
            pred_list = (y_predU_list['y_predU2'][:, k, np.newaxis],  y_predU_list['y_predU5'][:, k, np.newaxis],
                            y_predU_list['y_predU6'][:, k,
                                                    np.newaxis],  y_predU_list['y_predU7'][:, k, np.newaxis],
                            y_predU_list['y_predU8'][:, k,
                                                    np.newaxis],  y_predU_list['y_predU9'][:, k, np.newaxis],
                            y_predU_list['y_predU10'][:, k,
                                                    np.newaxis], y_predU_list['y_predU11'][:, k, np.newaxis],
                            y_predU_list['y_predU12'][:, k,
                                                    np.newaxis], y_predU_list['y_predU13'][:, k, np.newaxis],
                            y_predU_list['y_predU14'][:, k,
                                                    np.newaxis], y_predU_list['y_predU15'][:, k, np.newaxis],
                            y_predU_list['y_predU16'][:, k,
                                                    np.newaxis], y_predU_list['y_predU17'][:, k, np.newaxis],
                            y_predU_list['y_predU18'][:, k,
                                                    np.newaxis], y_predU_list['y_predU19'][:, k, np.newaxis],
                            y_predU_list['y_predU20'][:, k,
                                                    np.newaxis], y_predU_list['y_predU21'][:, k, np.newaxis],
                            y_predU_list['y_predU22'][:, k, np.newaxis], y_predU_list['y_predU23'][:, k, np.newaxis])

            X = np.concatenate(pred_list, axis=1)
            X_list.update({all_columns[f'{k}']: X})

        
        pred_list = {}
        for k in range(14):
            pos = pos_list[all_columns[f'{k}']]

            X_selected_features = X_list[all_columns[f'{k}']]
            X_selected_features = X_selected_features[:,pos==1]

            pred = np.mean(X_selected_features, axis=1)
            pred_list.update({all_columns[f'{k}']: list(pred)})


        all_preds = pd.DataFrame(pred_list)
        print(all_preds)
        return all_preds.values
#######################################################################


# Verificacion de Imagenes ############################################
def check_image(image, k):
    root = 'app'

    if image.filename == "":
        return render_template('/dashboard/chestXray_dashboard.html', upload=upload)

    if not allowed_image(image.filename):
        return render_template('/dashboard/chestXray_dashboard.html', upload=upload)
    else:
        filename = secure_filename(image.filename)

    file_ext = filename.split(".")[-1]
    file_name = filename.split(".")[0]
    
    if file_ext == 'png':
        filename = f'{k}.jpg'
        image.save(os.path.join(
            root, app.config['IMAGE_UPLOADS'], 'chestXray', filename))
    
    elif (file_ext == 'jpeg') or (file_ext == '.jpg'):
        filename = f'{k}.jpg'
        image.save(os.path.join(
            root, app.config['IMAGE_UPLOADS'], 'chestXray', filename))
    
    else: 
        filename = f'{k}.jpg'
        image.save(os.path.join(
            root, app.config['IMAGE_UPLOADS'], 'chestXray', filename))

    path = os.path.join(
        root, app.config['IMAGE_UPLOADS'], 'chestXray',  filename)

    return path, filename
#######################################################################


# Patologias ##########################################################
findings1 = {
    'ATELE': {'patologia': 'Atelectasia',
    
              'Descripción General': ('La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón. '
                                      'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
                                      'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
                                      'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
                                      'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
                                      'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.'),

              'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                           'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

              'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva). '
              'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares, '
              'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto '
              'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

              'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes: ',
                                     'Edad avanzada',
                                     'Una enfermedad que causa dificultad para tragar',
                                     'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                     'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                     'Cirugía abdominal o torácica reciente',
                                     'Anestesia general reciente',
                                     'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                     'Medicamentos que pueden causar respiración superficial',
                                     'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                     'Tabaquismo'
                                     ],

              'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes '
                                 'complicaciones pueden ser provocadas por atelectasia: ',
                                 ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                 'oxígeno a los sacos de aire(alvéolos).',
                                 'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                 'pulmón colapsado puede derivar en una infección.',
                                 'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                 'en bebés o en personas con enfermedad pulmonar.'
                                 ],

              'Prevención': ('La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia, '
                             'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una '
                             'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
                             'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después '
                             'de ciertas cirugías.'),

              'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la '
              'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se '
              'dificulta cada vez más, busca ayuda médica de emergencia.'
              },

    'CARD': {'patologia': 'Cardiomegalia',

             'Descripción General': 'La Cardiomegalia(corazón dilatado) no es una enfermedad, sino un signo de otra afección. El término "Cardiomegalia" se '
             'refiere a un corazón dilatado que se ve en cualquier prueba de diagnóstico por imágenes, como una radiografía de tórax. Luego se necesitan otras '
             'pruebas para diagnosticar la enfermedad que provoca la dilatación del corazón. '
             'Es posible que se presente una dilatación temporal del corazón debido a una situación de estrés para el organismo, como el embarazo, o debido a '
             'una enfermedad, como debilitamiento del músculo cardíaco, enfermedad de las arterias coronarias, problemas en las válvulas cardíacas o ritmos '
             'cardíacos anormales. Determinadas afecciones pueden hacer que los músculos del corazón se engrosen o que una de las cavidades se dilate, con lo '
             'cual aumenta el tamaño del corazón.',

             'Sintomas': ['En algunas personas, la cardiomegalia puede no causar signos ni síntomas. Otras personas pueden tener los siguientes signos y síntomas:',
                          'Dificultad para respirar',
                          'Ritmo cardíaco anormal(arritmia)',
                          'Hinchazón(edema)'
                          ],

             'Causas': 'El agrandamiento del corazón puede deberse a enfermedades que provocan que el corazón bombee con más fuerza de lo normal o que dañen el'
                       'músculo cardíaco. A veces, el corazón se agranda y se debilita por motivos desconocidos (idiopáticos).',

             'Factores de Riesgo': ['Puedes tener un mayor riesgo de presentar cardiomegalia si tienes alguno de los siguientes factores de riesgo:',
                                    'Presión arterial alta',
                                    'Antecedentes familiares de corazones agrandados o miocardiopatía',
                                    'Arterias bloqueadas en el corazón(enfermedad de las arterias coronarias)',
                                    'Enfermedad cardíaca congénita',
                                    'Enfermedad de las válvulas cardíacas',
                                    'Ataque cardíaco'
                                    ],

             'Complicaciones': ['El riesgo de sufrir complicaciones por cardiomegalia depende de la parte del corazón que se encuentre dilatada y de la causa.'
                                'Las complicaciones del corazón dilatado pueden comprender:',
                                'Insuficiencia cardíaca: Uno de los tipos más graves de dilatación del corazón, la dilatación del ventrículo izquierdo, aumenta'
                                'el riesgo de padecer insuficiencia cardíaca.',
                                'Coágulos sanguíneos: Tener cardiomegalia podría hacer que seas más propenso a la formación de coágulos de sangre en el revestimiento del corazón.',
                                'Soplo cardíaco: En las personas que tienen el corazón dilatado, es posible que dos de las cuatro válvulas del corazón —la válvula mitral y la'
                                'tricúspide— no cierren adecuadamente debido a que se dilatan, lo que provoca un reflujo de sangre.',
                                'Paro cardíaco y muerte súbita. Algunas formas de dilatación del corazón pueden provocar alteraciones en el ritmo de los latidos del corazón.'
                                ],

             'Prevención': 'Infórmale al médico si tienes antecedentes familiares de enfermedades que puedan provocar corazón dilatado, tal como la cardiomiopatía.'
             'Si la cardiomiopatía u otras enfermedades cardíacas se diagnostican de manera temprana, los tratamientos pueden evitar que empeore la enfermedad.'
             'Controlar los factores de riesgo de enfermedad de las arterias coronarias(el consumo de tabaco, la presión arterial alta, el colesterol alto y la'
             'diabetes) ayuda a reducir el riesgo de tener el corazón dilatado e insuficiencia cardíaca, pues reduce el riesgo de tener un ataque cardíaco.'
             'Puedes ayudar a reducir tus probabilidades de padecer insuficiencia cardíaca si sigues una dieta saludable y evitas el consumo excesivo de alcohol'
             'y el uso de drogas ilegales. Controlar la presión arterial alta con dieta, ejercicio y, posiblemente, medicamentos, también evita que muchas personas'
             'con el corazón dilatado padezcan insuficiencia cardíaca.',

             'Cuando debes consultar con un médico': 'Busca atención médica de urgencia si tienes algunos de los siguientes signos y síntomas, pues puede que estés'
             'teniendo un ataque cardíaco: Dolor en el pecho; Molestias en otras zonas de la parte superior del cuerpo, como uno de los brazos o ambos, la espalda,'
             'el cuello, la mandíbula o el estómago; Grave dificultad para respirar; Desmayo.'
             },

    'CONSO': {'patologia': 'Consolidación',
              'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
              'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
              'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
              'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
              'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
              'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

              'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                           'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

              'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
              'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
              'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
              'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

              'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                     'Edad avanzada',
                                     'Una enfermedad que causa dificultad para tragar',
                                     'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                     'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                     'Cirugía abdominal o torácica reciente',
                                     'Anestesia general reciente',
                                     'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                     'Medicamentos que pueden causar respiración superficial',
                                     'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                     'Tabaquismo'
                                     ],

              'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                 'complicaciones pueden ser provocadas por atelectasia',
                                 ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                 'oxígeno a los sacos de aire(alvéolos).',
                                 'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                 'pulmón colapsado puede derivar en una infección.',
                                 'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                 'en bebés o en personas con enfermedad pulmonar.'
                                 ],

              'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
              'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
              'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
              'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
              'de ciertas cirugías.',

              'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
              'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
              'dificulta cada vez más, busca ayuda médica de emergencia.'
              },

    'EDEMA': {'patologia': 'Edema',
              'Descripción General': 'El edema pulmonar es una enfermedad causada por el exceso de líquido presente en los pulmones. El líquido se acumula '
              'en las numerosas bolsas de aire de los pulmones y dificulta la respiración. En la mayoría de los casos, los problemas del corazón ocasionan '
              'edema pulmonar. Sin embargo, el líquido se puede acumular por otros motivos que incluyen la neumonía, la exposición a ciertas toxinas y '
              'medicamentos, el traumatismo en la pared torácica, y el visitar o hacer ejercicio en lugares de gran altitud. El edema que se manifiesta '
              'de manera repentina(edema pulmonar agudo) es una emergencia médica que se debe atender de inmediato. En algunas ocasiones, el edema pulmonar '
              'puede ser mortal pero el panorama mejora si se recibe atención rápida. El tratamiento del edema pulmonar varía según la causa pero, por lo '
              'general, incluye el suministro de oxígeno adicional y la administración de medicamentos.',

              'Sintomas': ['Según la causa, los signos y síntomas del edema pulmonar pueden aparecer de repente o desarrollarse con el tiempo. Existen '
                           'signos y síntomas de edema pulmonar repentino (agudo), por edema pulmonar de largo plazo (crónico) y por edema pulmonar de '
                           'gran altitud. Los sintomas mas comunes son:',
                           'La falta de aliento o la dificultad para respirar(disnea) excesiva que empeora con la actividad o al acostarse',
                           'Una sensación de asfixia o ahogamiento que empeora al acostarse',
                           'Sibilancias o jadeos para respirar',
                           'Piel fría y húmeda',
                           'Ansiedad, inquietud o sensación de aprehensión.',
                           'Una tos que produce expectoración espumosa que puede tener manchas de sangre',
                           'Labios azulados',
                           'Taquicardia o arritmia(palpitaciones)'
                           ],

              'Causas': 'Los pulmones contienen muchas bolsas de aire pequeñas y elásticas que se denominan alvéolos. Al respirar, estos absorben '
              'oxígeno y liberan dióxido de carbono. Por lo general, el intercambio de gases ocurre sin inconvenientes. Sin embargo, en ciertas '
              'circunstancias, los alvéolos se llenan de líquido en lugar de aire y evitan que el oxígeno se absorba en el torrente sanguíneo. '
              'Existen varios factores que pueden ocasionar la acumulación de líquido en los pulmones pero la mayoría tienen que ver con el '
              'corazón(edema pulmonar cardiogénico). El entender la relación que existe entre el corazón y los pulmones puede ayudar a explicar '
              'por qué ocurre esto.',

              'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                     'Edad avanzada',
                                     'Una enfermedad que causa dificultad para tragar',
                                     'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                     'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                     'Cirugía abdominal o torácica reciente',
                                     'Anestesia general reciente',
                                     'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                     'Medicamentos que pueden causar respiración superficial',
                                     'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                     'Tabaquismo'
                                     ],

              'Complicaciones': ['Si el edema pulmonar continúa, puede aumentar la presión en la arteria pulmonar (hipertensión pulmonar), y eventualmente '
                                 'el ventrículo derecho del corazón se debilita y comienza a fallar. El ventrículo derecho tiene una pared muscular mucho más delgada que '
                                 'el lado izquierdo del corazón, ya que está sujeto a menor presión para bombear la sangre hacia los pulmones. El aumento de presión se '
                                 'acumula en la aurícula derecha y después en varias partes del cuerpo, donde puede causar lo siguiente:',
                                 'Hinchazón abdominal y de las extremidades inferiores',
                                 'Acumulación de líquido en las membranas que rodean los pulmones(derrame pleural)',
                                 'Congestión e inflamación del hígado',

                                 ],

              'Prevención': 'La enfermedad cardiovascular es la principal causa del edema pulmonar. Con las siguientes recomendaciones, puedes reducir el '
              'riesgo de padecer muchos tipos de problemas cardíacos:',

              'Cuando debes consultar con un médico': 'El edema pulmonar que aparece repentinamente es potencialmente mortal. Llama a la atención médica de '
              'emergencia si presentas alguno de los siguientes signos y síntomas agudos: Dificultad para respirar, especialmente si es repentina '
              'Dificultad para respirar o sensación de sofocación(disnea), '
              'Un sonido burbujeante, sibilante o jadeante al respirar, '
              'Expectoración espumosa y rosa al toser, '
              'Dificultad respiratoria junto con sudoración profusa, '
              'Piel azulada o grisácea, '
              'Desorientación, '
              'Una disminución significativa de la presión arterial que provoca aturdimiento, mareos, debilidad o sudoración, '
              'Un empeoramiento repentino de cualquiera de los síntomas asociados con edema pulmonar crónico o edema pulmonar de gran altitud.'
              },


    'EFFUS': {'patologia': 'Derrame Pleural',
              'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
              'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
              'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
              'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
              'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
              'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

              'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                           'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

              'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
              'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
              'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
              'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

              'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                     'Edad avanzada',
                                     'Una enfermedad que causa dificultad para tragar',
                                     'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                     'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                     'Cirugía abdominal o torácica reciente',
                                     'Anestesia general reciente',
                                     'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                     'Medicamentos que pueden causar respiración superficial',
                                     'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                     'Tabaquismo'
                                     ],

              'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                 'complicaciones pueden ser provocadas por atelectasia',
                                 ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                 'oxígeno a los sacos de aire(alvéolos).',
                                 'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                 'pulmón colapsado puede derivar en una infección.',
                                 'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                 'en bebés o en personas con enfermedad pulmonar.'
                                 ],

              'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
              'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
              'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
              'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
              'de ciertas cirugías.',

              'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
              'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
              'dificulta cada vez más, busca ayuda médica de emergencia.'
              },


    'PNEUM': {'patologia': 'Neumonía',
              'Descripción General': 'La neumonía es una infección que inflama los sacos aéreos de uno o ambos pulmones. Los sacos aéreos se pueden llenar '
              'de líquido o pus(material purulento), lo que provoca tos con flema o pus, fiebre, escalofríos y dificultad para respirar. Diversos '
              'microrganismos, como bacterias, virus y hongos, pueden provocar neumonía. La neumonía puede variar en gravedad desde suave a potencialmente '
              'mortal. Es más grave en bebés y niños pequeños, personas mayores a 65 años, y personas con problemas de salud o sistemas inmunitarios debilitados.',

              'Sintomas': ['Los signos y síntomas de la neumonía varían de moderados a graves y dependen de varios factores, como el tipo de germen que causó '
                           'la infección, tu edad y tu salud en general. Los signos y síntomas moderados suelen ser similares a los de un resfrío o una gripe, '
                           'pero duran más tiempo. Los signos y síntomas de la neumonía pueden incluir lo siguiente:',
                           'Dolor en el pecho al respirar o toser',
                           'Desorientación o cambios de percepción mental(en adultos de 65 años o más)',
                           'Tos que puede producir flema',
                           'Fatiga',
                           'Fiebre, transpiración y escalofríos con temblor',
                           'Temperatura corporal más baja de lo normal(en adultos mayores de 65 años y personas con un sistema inmunitario débil)',
                           'Náuseas, vómitos o diarrea',
                           'Dificultad para respirar'
                           ],

              'Causas': 'Son varios los gérmenes que pueden causar neumonía. Los más frecuentes son las bacterias y los virus que se encuentran en el aire '
                        'que respiramos. Generalmente, el cuerpo evita que estos gérmenes infecten los pulmones. Sin embargo, a veces, estos gérmenes pueden '
                        'ser más poderosos que tu sistema inmunitario, incluso cuando tu salud en general es buena. La neumonía se clasifica de acuerdo con '
                        'el tipo de germen que la causa y el lugar donde tienes la infección.',

              'Factores de Riesgo': ['La neumonía puede afectar a cualquiera. Pero los dos grupos de edades que presentan el mayor riesgo de padecerla '
                                     'son los siguientes:',
                                     'Estar hospitalizado. Tienes un mayor riesgo de contraer neumonía si te encuentras en la unidad de cuidados '
                                     'intensivos de un hospital, especialmente, si estás conectado a una máquina que te ayuda a respirar(ventilador).',
                                     'Enfermedad crónica. Eres más propenso a contraer neumonía si tienes asma, enfermedad pulmonar obstructiva '
                                     'crónica o una enfermedad cardíaca.',
                                     'Fumar. El fumar daña las defensas naturales que tu cuerpo presenta contra las bacterias y los virus que causan neumonía.',
                                     'Sistema inmunitario debilitado o suprimido. Las personas que tienen VIH/SIDA, que se han sometido a un '
                                     'trasplante de órganos o que reciben quimioterapia o esteroides a largo plazo están en riesgo.',
                                     ],

              'Complicaciones': ['Incluso habiendo recibido tratamiento, algunas personas que tienen neumonía, especialmente aquellos que se encuentran '
                                 'en los grupos de alto riesgo, pueden experimentar complicaciones, incluidas las siguiente:'
                                 'Bacterias en el torrente sanguíneo(bacteriemia). Las bacterias que ingresan en el torrente sanguíneo desde los pulmones '
                                 'pueden propagar la infección a otros órganos y, potencialmente, provocar una insuficiencia orgánica.',
                                 'Dificultad para respirar. Si la neumonía es grave o si tienes enfermedades pulmonares crónicas ocultas, posiblemente tengas '
                                 'problemas para obtener suficiente oxígeno al respirar. Es posible que debas hospitalizarte y utilizar un respirador '
                                 'artificial(ventilador) hasta que tus pulmones sanen.',
                                 'Acumulación de líquido alrededor de los pulmones(derrame pleural). La neumonía puede causar la acumulación de líquido en '
                                 'el fino espacio que hay entre las capas de tejido que recubren los pulmones y la cavidad torácica(pleura). Si el fluido se '
                                 'infecta, es posible que deban drenarlo a través de una sonda pleural o extraerlo mediante una cirugía.',
                                 'Absceso pulmonar. Un absceso tiene lugar si se forma pus en una cavidad en el pulmón. Normalmente, los abscesos se tratan '
                                 'con antibióticos. A veces, se necesita una cirugía o un drenaje con una aguja larga o una sonda que se coloca en el absceso para extraer el pus.'
                                 ],

              'Prevención': 'Para contribuir a prevenir la neumonía: Vacúnate: Existen vacunas para prevenir algunos tipos de neumonía y la gripe; '
              'Asegúrate de que los niños se vacunen: Los médicos recomiendan una vacuna para la neumonía diferente para niños menores de 2 años '
              'y para niños de 2 a 5 años que son particularmente propensos a contraer la enfermedad neumocócica; Practica una buena higiene: Para '
              'protegerte de las infecciones respiratorias que a menudo derivan en neumonía; No fumes: El tabaquismo daña las defensas naturales '
              'que protegen a tus pulmones de las infecciones respiratorias; Mantén fuerte tu sistema inmunitario. Duerme lo suficiente, ejercítate '
              'regularmente y lleva una dieta saludable.',

              'Cuando debes consultar con un médico': 'Consulta con tu médico si tienes dificultad para respirar, dolor en el pecho, fiebre persistente de 39 ºC '
              'o superior, o tos persistente, sobre todo si tienes tos con pus. Es muy importante que las personas que pertenecen a los siguientes grupos '
              'de riesgo consulten al médico: Adultos mayores de 65 años, niños menores de 2 años con signos y síntomas, personas con alguna afección de salud '
              'no diagnosticada o con el sistema inmunitario debilitado, personas que reciben quimioterapia o toman medicamentos que inhiben el sistema inmunitario'
              },

    'PNTHO': {'patologia': 'Neumotoráx',
              'Descripción General': 'Neumotórax es un colapso pulmonar. El neumotórax se produce cuando el aire se filtra dentro del espacio que se '
              'encuentra entre los pulmones y la pared torácica. El aire hace presión en la parte externa del pulmón y lo hace colapsar. El neumotórax puede '
              'ser un colapso pulmonar completo o un colapso de solo una parte del pulmón. El neumotórax puede ser provocado por una contusión o una lesión '
              'penetrante en el pecho, por determinados procedimientos médicos o daño provocado por una enfermedad pulmonar oculta. O bien, puede ocurrir sin '
              'un motivo evidente. Los síntomas, generalmente, comprenden dolor repentino en el pecho y dificultad para respirar. En algunas ocasiones, un '
              'colapso pulmonar puede ser un evento que pone en riesgo la vida. En general, el tratamiento del neumotórax implica introducir una aguja o un '
              'tubo en el pecho entre las costillas para eliminar el exceso de aire. Sin embargo, un pequeño neumotórax puede curarse por sí solo.',

              'Sintomas': ['Los principales síntomas del neumotórax son:', 'Dolor de tórax repentino', 'Dificultad para respirar.'],

              'Causas': 'Un neumotórax puede originarse por: Lesión en el pecho: Cualquier contusión o lesión penetrante en el pecho puede provocar '
                        'el colapso pulmonar. Algunas lesiones pueden producirse durante agresiones físicas o accidentes de automóvil, mientras '
                        'que otras pueden producirse por accidente durante procedimientos médicos que implican la inserción de una aguja en el pecho; '
                        'Enfermedad pulmonar: Es más probable que el tejido pulmonar dañado colapse. El daño pulmonar puede originarse en muchos '
                        'tipos de enfermedades ocultas, entre ellas, la enfermedad pulmonar obstructiva crónica, la fibrosis quística y la neumonía; '
                        'Ampollas de aire rotas: Se pueden manifestar pequeñas ampollas de aire en la parte superior de los pulmones. Estas ampollas '
                        'a veces revientan, dejando que el aire se filtre en el espacio que rodea los pulmones; Ventilación mecánica: Se puede producir '
                        'un tipo grave de neumotórax en personas que necesitan asistencia mecánica para respirar. El respirador puede crear un '
                        'desequilibrio de presión de aire dentro del pecho. El pulmón puede colapsar por completo.',

              'Factores de Riesgo': ['En general, los hombres tienen mayor probabilidad de tener neumotórax que las mujeres. El tipo de neumotórax '
                                     'provocado por ampollas de aire rotas tiene más probabilidad de ocurrir en personas entre los 20 y los 40 años, '
                                     'en especial, si la persona es muy alta y tiene bajo peso. Entre los factores de riesgo de un neumotórax, se encuentran:',
                                     'Tabaquismo: El riesgo se incrementa con la cantidad de tiempo y con la cantidad de cigarrillos fumados, incluso '
                                     'cuando no hay enfisema.',
                                     'La genética: Ciertos tipos de neumotórax tienden a ser hereditarios.',
                                     'Enfermedad pulmonar: Tener una enfermedad pulmonar de fondo, en especial, la enfermedad pulmonar obstructiva crónica, '
                                     'hace más probable la aparición de un colapso pulmonar.',
                                     'Ventilación mecánica: Las personas que necesitan ventilación mecánica para asistir su respiración corren '
                                     'un mayor riesgo de padecer un neumotórax.',
                                     'Neumotórax previo: Cualquier persona que haya tenido un neumotórax corre mayor riesgo de tener otro.'
                                     ],

              'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                 'complicaciones pueden ser provocadas por atelectasia',
                                 ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                 'oxígeno a los sacos de aire(alvéolos).',
                                 'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                 'pulmón colapsado puede derivar en una infección.',
                                 'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                 'en bebés o en personas con enfermedad pulmonar.'
                                 ],

              'Prevención': 'Muchas personas que han tenido un neumotórax pueden tener otro, habitualmente después de uno a dos años del primero. '
                            'A veces, puede seguir filtrándose aire si la abertura en el pulmón no se cierra. Es posible que sea necesario hacer '
                            'una cirugía para cerrar la fuga de aire.',

              'Cuando debes consultar con un médico': 'Estos síntomas pueden ser causados por diversos problemas de salud y algunos de ellos pueden '
              'ser potencialmente mortales, de manera que busca atención médica. Si el dolor de tórax es intenso o respirar se torna cada '
              'vez más difícil, busca atención de urgencia de inmediato.'
              },

    'FRACT': {'patologia': 'Fractura',
              'Descripción General': 'La fractura de costilla es una lesión frecuente que ocurre cuando uno de los huesos de la caja torácica se '
              'quiebra o se fisura. La causa más frecuente es el traumatismo de pecho, como una caída, un accidente automovilístico o impacto durante '
              'la práctica de deportes de contacto. En muchos casos, las costillas fracturadas en realidad están solamente fisuradas. Si bien son dolorosas, '
              'las fisuras de costilla no son potencialmente tan peligrosas como las costillas que se han fracturado en partes pequeñas. Los bordes '
              'irregulares de un hueso fracturado pueden dañar los vasos sanguíneos u órganos internos principales, como los pulmones. Un adecuado control '
              'del dolor es importante para que puedas continuar respirando profundamente y evitar complicaciones pulmonares, como la neumonía.',

              'Sintomas': ['El dolor asociado con una fractura de costilla generalmente se produce o empeora cuando:',
                           'Respiras profundamente',
                           'Presionas sobre la zona lesionada',
                           'Inclinas o giras el cuerpo'
                           ],

              'Causas': 'La causa más frecuente de las fracturas de costillas son los impactos directos, como aquellos de los accidentes automovilísticos, '
                        'las caídas, el maltrato infantil o los deportes de contacto. Las costillas también pueden fracturarse por traumatismos reiterados '
                        'de deportes como el golf y el remo o debido a la tos intensa y prolongada.',

              'Factores de Riesgo': ['Los siguientes factores pueden aumentar el riesgo de fracturarte una costilla:',
                                     'Osteoporosis. Tener esta enfermedad en la que los huesos pierden densidad te hace más susceptible a fracturarte un hueso.',
                                     'Práctica de deportes. Practicar deportes de contacto, como hockey o fútbol, aumenta el riesgo de sufrir traumatismo en el pecho.',
                                     'Lesión cancerosa en una costilla. Una lesión cancerosa puede debilitar al hueso, y lo hace más susceptible a sufrir fracturas.',
                                     'Una costilla fracturada puede dañar los vasos sanguíneos y los órganos internos. El riesgo aumenta con la cantidad de '
                                     'costillas fracturadas. Las complicaciones varían según qué costillas te fractures.',

                                     ],

              'Complicaciones': ['Algunas de las posibles complicaciones son las siguientes:',
                                 'Rotura o perforación de la aorta. El borde filoso de una fractura en una de las tres primeras costillas en la parte '
                                 'superior de la caja torácica podría romper la aorta u otro vaso sanguíneo importante.',
                                 'Perforación del pulmón. El borde irregular de una costilla media fracturada puede perforar un pulmón y hacer que colapse.',
                                 'Desgarro del bazo, del hígado o de los riñones. Las dos costillas inferiores en raras ocasiones se fracturan, porque '
                                 'tienen más flexibilidad que las costillas superiores y medias, que se encuentras fijadas al esternón. Pero si te '
                                 'fracturas una costilla inferior, los bordes rotos pueden provocar daños graves en el bazo, en el hígado o en un riñón.'
                                 ],

              'Prevención': 'Las siguientes medidas pueden ayudarte a evitar una fractura de costilla: Protégete de las lesiones atléticas; '
              'Usa equipo protector cuando hagas deportes de contacto; Reduce el riesgo de caídas en el hogar; Ordena el desorden de los pisos de tu hogar '
              'y limpia de inmediato todo lo que caiga en el suelo, usa tapete de goma en la ducha, mantén una buena iluminación en el hogar y coloca '
              'refuerzos antideslizantes en las alfombras y tapetes. Fortalece los huesos. Obtener suficiente calcio y vitamina D en tu dieta es '
              'importante para mantener los huesos fuertes.',

              'Cuando debes consultar con un médico': 'Consulta con el médico si aparece un punto muy sensible en la zona de las costillas después '
              'de un traumatismo, o si tienes dificultad para respirar o dolor al respirar de manera profunda. Busca atención médica de inmediato si sientes '
              'presión, inflamación o dolor opresivo en el centro del pecho que dura más de unos pocos minutos o dolor que se extiende más allá del pecho y '
              'llega al hombro o al brazo. Estos síntomas pueden indicar un ataque cardíaco.'
              }
}
#######################################################################







#######################################################################
findings2 = {'ENLAR': {'patologia': 'Cardiomediastino Agrandado',
                       'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
                       'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
                       'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
                       'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
                       'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
                       'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

                       'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                                    'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

                       'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
                       'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
                       'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
                       'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

                       'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                              'Edad avanzada',
                                              'Una enfermedad que causa dificultad para tragar',
                                              'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                              'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                              'Cirugía abdominal o torácica reciente',
                                              'Anestesia general reciente',
                                              'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                              'Medicamentos que pueden causar respiración superficial',
                                              'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                              'Tabaquismo'
                                              ],

                       'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                          'complicaciones pueden ser provocadas por atelectasia',
                                          ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                          'oxígeno a los sacos de aire(alvéolos).',
                                          'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                          'pulmón colapsado puede derivar en una infección.',
                                          'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                          'en bebés o en personas con enfermedad pulmonar.'
                                          ],

                       'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
                       'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
                       'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
                       'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
                       'de ciertas cirugías.',

                       'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
                       'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
                       'dificulta cada vez más, busca ayuda médica de emergencia.'
                       },

             'LESIO': {'patologia': 'Lesión Pulmonar',
                       'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
                       'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
                       'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
                       'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
                       'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
                       'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

                       'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                                    'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

                       'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
                       'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
                       'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
                       'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

                       'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                              'Edad avanzada',
                                              'Una enfermedad que causa dificultad para tragar',
                                              'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                              'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                              'Cirugía abdominal o torácica reciente',
                                              'Anestesia general reciente',
                                              'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                              'Medicamentos que pueden causar respiración superficial',
                                              'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                              'Tabaquismo'
                                              ],

                       'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                          'complicaciones pueden ser provocadas por atelectasia',
                                          ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                          'oxígeno a los sacos de aire(alvéolos).',
                                          'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                          'pulmón colapsado puede derivar en una infección.',
                                          'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                          'en bebés o en personas con enfermedad pulmonar.'
                                          ],

                       'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
                       'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
                       'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
                       'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
                       'de ciertas cirugías.',

                       'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
                       'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
                       'dificulta cada vez más, busca ayuda médica de emergencia.'
                       },

             'OPACI': {'patologia': 'Opacidad Pulmonar',
                       'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
                       'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
                       'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
                       'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
                       'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
                       'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

                       'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                                    'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

                       'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
                       'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
                       'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
                       'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

                       'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                              'Edad avanzada',
                                              'Una enfermedad que causa dificultad para tragar',
                                              'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                              'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                              'Cirugía abdominal o torácica reciente',
                                              'Anestesia general reciente',
                                              'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                              'Medicamentos que pueden causar respiración superficial',
                                              'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                              'Tabaquismo'
                                              ],

                       'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                          'complicaciones pueden ser provocadas por atelectasia',
                                          ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                          'oxígeno a los sacos de aire(alvéolos).',
                                          'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                          'pulmón colapsado puede derivar en una infección.',
                                          'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                          'en bebés o en personas con enfermedad pulmonar.'
                                          ],

                       'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
                       'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
                       'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
                       'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
                       'de ciertas cirugías.',

                       'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
                       'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
                       'dificulta cada vez más, busca ayuda médica de emergencia.'
                       },
             'POTHE': {'patologia': 'Otro Pleural',
                       'Descripción General': 'La atelectasia es un colapso completo o parcial del pulmón entero o de una parte(lóbulo) del pulmón.'
                       'Se produce cuando las pequeñas bolsas de aire(alvéolos) que forman los pulmones se desinflan o posiblemente se llenan de líquido. '
                       'La atelectasia es una de las complicaciones respiratorias más frecuentes después de una cirugía. También es una posible complicación '
                       'de otros problemas respiratorios, como la fibrosis quística, los tumores de pulmón, las lesiones en el tórax, el líquido en los '
                       'pulmones y la debilidad respiratoria. También puedes tener atelectasia si inhalas un objeto extraño. La atelectasia puede dificultar '
                       'la respiración, especialmente si ya tienes una enfermedad pulmonar. El tratamiento depende de la causa y la gravedad del colapso.',

                       'Sintomas': ['Es posible que no haya signos ni síntomas evidentes de la atelectasia. Sin embargo, cuando estos aparecen, suelen ser los siguientes:',
                                    'Dificultad para respirar', 'Respiración agitada y superficial', 'Sibilancias', 'Tos'],

                       'Causas': 'La atelectasia ocurre por una vía respiratoria bloqueada(obstructiva) o por presión externa al pulmón(no obstructiva).'
                       'La anestesia general es una causa común de atelectasia. Cambia el ritmo regular de respiración y afecta el intercambio de gases pulmonares,'
                       'lo que puede hacer que los sacos de aire(alvéolos) se desinflen. Casi todas las personas que se someten a una cirugía mayor desarrollan cierto'
                       'grado de atelectasia. Suele ocurrir después de una cirugía de derivación coronaria.',

                       'Factores de Riesgo': ['Los factores que aumentan las probabilidades de desarrollar atelectasia son los siguientes',
                                              'Edad avanzada',
                                              'Una enfermedad que causa dificultad para tragar',
                                              'Reposo absoluto en cama con cambios de posición poco frecuentes',
                                              'Enfermedad pulmonar, como asma, EPOC, bronquiectasia o fibrosis quística',
                                              'Cirugía abdominal o torácica reciente',
                                              'Anestesia general reciente',
                                              'Músculos respiratorios débiles debido a distrofia muscular, lesión de la médula espinal u otra enfermedad neuromuscular',
                                              'Medicamentos que pueden causar respiración superficial',
                                              'Dolor o lesión que puede producir dolor al toser o causar respiración superficial, incluido dolor de estómago o fractura de costilla',
                                              'Tabaquismo'
                                              ],

                       'Complicaciones': ['Por lo general, una pequeña zona de atelectasia, en especial en los adultos, es tratable. Las siguientes'
                                          'complicaciones pueden ser provocadas por atelectasia',
                                          ' Bajo nivel de oxígeno en sangre(hipoxemia). La atelectasia obstaculiza la capacidad de los pulmones de llevar'
                                          'oxígeno a los sacos de aire(alvéolos).',
                                          'Neumonía. Tu riesgo de contraer neumonía continúa hasta que la atelectasia haya desaparecido. La mucosidad en un'
                                          'pulmón colapsado puede derivar en una infección.',
                                          'Insuficiencia respiratoria. La pérdida de un lóbulo o de un pulmón entero puede ser potencialmente fatal, en especial'
                                          'en bebés o en personas con enfermedad pulmonar.'
                                          ],

                       'Prevención': 'La atelectasia en los niños a menudo es provocada por un bloqueo de las vías respiratorias. A fin de reducir el riesgo de atelectasia,'
                       'mantén los objetos pequeños fuera del alcance de los niños. En el caso de los adultos, la atelectasia ocurre con mayor frecuencia después de una'
                       'cirugía mayor. Si tienes una cirugía programada, habla con el médico sobre las estrategias para reducir los riesgos de atelectasia. Algunas '
                       'investigaciones sugieren que ciertos ejercicios de respiración y entrenamiento de los músculos pueden disminuir el riesgo de atelectasia después'
                       'de ciertas cirugías.',

                       'Cuando debes consultar con un médico': 'Siempre busca atención médica inmediata si tienes problemas para respirar. Otras enfermedades, además de la'
                       'atelectasia, pueden ocasionar dificultades respiratorias y requieren un diagnóstico certero y tratamiento inmediato. Si sientes que tu respiración se'
                       'dificulta cada vez más, busca ayuda médica de emergencia.'
                       },



             }
#######################################################################
