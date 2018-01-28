# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 22:22:45 2017

@author: Romain
"""

from lxml import etree

from grapa.graph import Graph
from grapa.curve import Curve
from grapa.mathModule import stringToVariable


class GraphXML(Graph):
    
    FILEIO_GRAPHTYPE = 'Undetermined measurement type'
    
    @classmethod
    def isFileReadable(cls, fileName, fileExt, **kwargs):
        if fileExt in ['.xml']:
            return True
        return False

    
    def readDataFromFile(self, attributes, **kwargs):
        """
        Will read output of GraphIO.exportXML: interpret and import content of
        xml file.
        Quite basic at the moment. Did that more for fun than for need.
        """
        tree = etree.parse(self.filename)
        root = tree.getroot()
        
        def elementToAttributes(root, element, where):
            content = root.find(element)
            if content is not None:
                keys = content.keys()
                for key in keys:
                    where.update({key: stringToVariable(content.get(key))})
    
        # interpret headers
        for container in ['headers', 'sampleInfo', 'graphInfo', 'dummy']:
            if hasattr(self, container):
                elementToAttributes(root, container, getattr(self, container))
        # interpret curves
        c = 0
        while 1:
            curve = root.find('curve'+str(c))
            if curve is None:
                break
            data = curve.find('data')
            x = data.find('x')
            y = data.find('y')
            x = stringToVariable('['+x.text+']') if x is not None else []
            y = stringToVariable('['+y.text+']') if x is not None else []
            crv = Curve([x, y], {})
            elementToAttributes(curve, 'attributes', crv)
            # cast Curbe to correct child class if parameter provided
            if crv.getAttribute('curve') not in ['', 'curve', 'curvexy']:
                new = crv.castCurve(crv.getAttribute('curve'))
                if isinstance(new, Curve):
                    crv = new
            self.append(crv)
            c += 1        
        
    """
    @classmethod
    def importEtree(cls):
        #Some variations how to import etree (http://lxml.de/tutorial.html)
        #Don't know if this could be any useful
        try:
            from lxml import etree
            #print("running with lxml.etree")
        except ImportError:
            try:
                # Python 2.5
                import xml.etree.cElementTree as etree
                #print("running with cElementTree on Python 2.5+")
            except ImportError:
                try:
                    # Python 2.5
                    import xml.etree.ElementTree as etree
                    #print("running with ElementTree on Python 2.5+")
                except ImportError:
                    try:
                        # normal cElementTree install
                        import cElementTree as etree
                        #print("running with cElementTree")
                    except ImportError:
                        try:
                            # normal ElementTree install
                            import elementtree.ElementTree as etree
                            #print("running with ElementTree")
                        except ImportError:
                            print("GraphXML: Failed to import ElementTree from any known place")
                            raise ImportError
      """