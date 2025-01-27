PROMPT_TEMPLATE = """
Du bist ein erfahrener Radiologe.  
Deine Aufgabe ist es, strukturierte Informationen aus einem Freitext-Radiologiebericht zu extrahieren.  
Diese strukturierten Informationen sind als Fakten organisiert, die ununterbrochene Textabschnitte darstellen.  
Du erhältst eine Vorlage mit dem Namen einer spezifischen Faktenklasse und den Attributen, die ausgefüllt werden müssen.  
Gib nur die angegebenen Instanzen der Faktenklasse zurück. Es kann mehrere Fakten im selben Dokument geben.  
Fülle die vordefinierten Attribute innerhalb eines Faktenabschnitts mit exakt demselben Text aus dem Bericht aus.  
Wenn Attribute im Text nicht vorhanden sind, lasse die Zeichenkette leer.  
Gib ein JSON-Objekt gemäß der bereitgestellten Vorlage zurück.  

Hier ein paar Beispiele: 

{EXAMPLES}

Dies ist der Freitext-Radiologiebericht, der strukturiert werden soll:

{DOCUMENT}

Dies ist die auszufüllende Berichts-Vorlage:

{REPORT_TEMPLATE}

Denke Schritt für Schritt. 
"""