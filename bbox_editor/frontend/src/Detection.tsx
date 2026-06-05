import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { useEffect, useState } from "react"
import { ChakraProvider, SimpleGrid, Box, Flex, Center, Button, Text } from '@chakra-ui/react'

import useImage from 'use-image';

import ThemeSwitcher from './ThemeSwitcher'

import BBoxCanvas from "./BBoxCanvas";

export interface PythonArgs {
  image_url: string,
  image_size: number[],
  label_list: string[],
  bbox_info: any[],
  color_map: any,
  line_width: number,
  use_space: boolean,
  token: number
}

const Detection = ({ args, theme }: ComponentProps) => {
  const {
    image_url,
    image_size,
    label_list,
    bbox_info,
    color_map,
    line_width,
    use_space,
    token
  }: PythonArgs = args

  const params = new URLSearchParams(window.location.search);
  const baseUrl = params.get('streamlitUrl')

  let imageUrl: string
  if (baseUrl) {
    const url = new URL(baseUrl)
    const cleanPath = url.pathname.endsWith('/')
      ? url.pathname
      : url.pathname.substring(0, url.pathname.lastIndexOf('/') + 1)
    imageUrl = url.origin + cleanPath + image_url.substring(1)
  } else {
    imageUrl = image_url
  }

  const [image] = useImage(imageUrl)

  const [rectangles, setRectangles] = React.useState(
    bbox_info.map((bb, i) => ({
      x: bb.bbox[0],
      y: bb.bbox[1],
      width: bb.bbox[2],
      height: bb.bbox[3],
      label: bb.label,
      stroke: color_map[bb.label],
      id: 'bbox-' + i
    }))
  );
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [label, setLabel] = useState(label_list[0])

  const handleClassSelect = (l: string) => {
    setLabel(l)
    if (selectedId !== null) {
      const rects = rectangles.slice()
      for (let i = 0; i < rects.length; i++) {
        if (rects[i].id === selectedId) {
          rects[i].label = l
          rects[i].stroke = color_map[l]
        }
      }
      setRectangles(rects)
    }
  }

  const [scale, setScale] = useState(1.0)
  useEffect(() => {
    const resizeCanvas = () => {
      const isMobile = window.innerWidth < 768
      // Sur mobile le panneau de classe est en dessous → le canvas peut utiliser ~93 % de la largeur.
      // Sur desktop il est à droite → on laisse ~78 % pour le canvas.
      const ratio = isMobile ? 0.93 : 0.78
      const scale_ratio = window.innerWidth * ratio / image_size[0]
      setScale(Math.min(scale_ratio, 1.0))
      const imageHeight = image_size[1] * Math.min(scale_ratio, 1.0)
      // Sur mobile on ajoute la hauteur du panneau de classe empilé en dessous (~160 px)
      const extraHeight = isMobile ? 160 : 0
      Streamlit.setFrameHeight(Math.max(imageHeight + extraHeight, 200))
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas()
    return () => { window.removeEventListener('resize', resizeCanvas); }
  }, [image_size])

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (use_space && event.key === ' ') {
        Streamlit.setComponentValue({
          token,
          bboxes: rectangles.map((rect) => ({
            bbox: [rect.x, rect.y, rect.width, rect.height],
            label_id: label_list.indexOf(rect.label),
            label: rect.label
          }))
        })
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => { window.removeEventListener('keydown', handleKeyPress); };
  }, [rectangles, token]);

  const sendValue = () => {
    Streamlit.setComponentValue({
      token,
      bboxes: rectangles.map((rect) => ({
        bbox: [rect.x, rect.y, rect.width, rect.height],
        label_id: label_list.indexOf(rect.label),
        label: rect.label
      }))
    })
  }

  return (
    <ChakraProvider>
      <ThemeSwitcher theme={theme}>
        <Center>
          <Flex
            direction={{ base: 'column', md: 'row' }}
            gap={3}
            width="100%"
            align={{ base: 'stretch', md: 'flex-start' }}
          >
            <Box>
              <BBoxCanvas
                rectangles={rectangles}
                selectedId={selectedId}
                scale={scale}
                setSelectedId={setSelectedId}
                setRectangles={setRectangles}
                setLabel={setLabel}
                color_map={color_map}
                label={label}
                image={image}
                image_size={image_size}
                strokeWidth={line_width}
              />
            </Box>
            <Box minW={{ md: '140px' }}>
              <Text fontSize='sm' mb={2}>Classe</Text>
              <SimpleGrid columns={{ base: 4, md: 1 }} gap={2}>
                {label_list.map((l) => {
                  const isActive = label === l
                  return (
                    <Button
                      key={l}
                      size='sm'
                      bg={isActive ? color_map[l] : 'transparent'}
                      borderColor={color_map[l]}
                      borderWidth='2px'
                      color={isActive ? 'white' : color_map[l]}
                      _hover={{ bg: color_map[l], color: 'white' }}
                      onClick={() => handleClassSelect(l)}
                      width='100%'
                    >
                      {l}
                    </Button>
                  )
                })}
              </SimpleGrid>
              <Text fontSize='xs' color='gray.500' mt={2}>
                Clic droit sur une bbox pour la supprimer
              </Text>
              <Button mt={3} width="100%" onClick={sendValue}>Valider</Button>
            </Box>
          </Flex>
        </Center>
      </ThemeSwitcher>
    </ChakraProvider>
  )
}

export default withStreamlitConnection(Detection)
