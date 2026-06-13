import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { useEffect, useState } from "react"
import { ChakraProvider, SimpleGrid, Box, Flex, Center, Button, Text, Slider, SliderTrack, SliderFilledTrack, SliderThumb } from '@chakra-ui/react'

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

// Largeur fixe du panneau de contrôle (boutons de classe + Valider) en px
const SIDE_PANEL_W = 230
// gap entre canvas et panneau en px
const GAP = 16

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
  const [bboxOpacity, setBboxOpacity] = useState(1.0)
  const [bboxStroke,  setBboxStroke]  = useState(line_width)

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
      // Sur desktop : on soustrait le panneau latéral fixe + le gap
      const reserved = isMobile ? 0 : SIDE_PANEL_W + GAP
      const availableWidth = Math.max(100, window.innerWidth - reserved)
      const scale_ratio = availableWidth / image_size[0]
      // Pas d'upscaling : l'image source est déjà à 1280 px max, pas besoin d'agrandir
      setScale(Math.min(scale_ratio, 1.0))
      const imageHeight = image_size[1] * Math.min(scale_ratio, 1.0)
      // Sur mobile le panneau de contrôle est empilé en dessous (~160 px)
      const extraHeight = isMobile ? 160 : 0
      Streamlit.setFrameHeight(Math.max(imageHeight + extraHeight + 10, 200))
    }

    window.addEventListener('resize', resizeCanvas)
    resizeCanvas()
    return () => { window.removeEventListener('resize', resizeCanvas) }
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
            gap={`${GAP}px`}
            width={{ base: '100%', md: 'auto' }}
            align={{ base: 'stretch', md: 'center' }}
          >
            {/* Canvas — dimensionné précisément par resizeCanvas */}
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
                strokeWidth={bboxStroke}
                opacity={bboxOpacity}
              />
            </Box>

            {/* Panneau de contrôle — largeur fixe sur desktop */}
            <Box
              flexShrink={0}
              width={{ base: '100%', md: `${SIDE_PANEL_W}px` }}
              pl={{ md: 2 }}
            >
              <Text fontSize='sm' fontWeight='semibold' color='gray.600' mb={3}>
                Classe
              </Text>

              {/* Boutons de classe + Valider côte à côte */}
              <Flex gap={3} align="stretch">
                <SimpleGrid columns={{ base: 2, md: 1 }} gap={3} flex="1">
                  {label_list.map((l) => {
                    const isActive = label === l
                    return (
                      <Button
                        key={l}
                        size='md'
                        bg={isActive ? color_map[l] : 'transparent'}
                        borderColor={color_map[l]}
                        borderWidth='2px'
                        color={isActive ? 'white' : color_map[l]}
                        _hover={{ bg: color_map[l], color: 'white' }}
                        onClick={() => handleClassSelect(l)}
                        width='100%'
                        fontWeight={isActive ? 'bold' : 'normal'}
                      >
                        {l}
                      </Button>
                    )
                  })}
                </SimpleGrid>

                {/* Valider — vert, pleine hauteur du bloc de classe */}
                <Button
                  colorScheme='green'
                  onClick={sendValue}
                  height="auto"
                  minH="100%"
                  px={5}
                  fontSize='md'
                  fontWeight='bold'
                  flexShrink={0}
                  boxShadow='md'
                >
                  Valider
                </Button>
              </Flex>

              {/* Slider opacité */}
              <Flex align="center" gap={2} mt={4}>
                <Text fontSize='xs' color='gray.500' minW="50px">Opacité</Text>
                <Slider flex="1" value={bboxOpacity} min={0.1} max={1} step={0.05}
                        onChange={(v) => setBboxOpacity(v)}>
                  <SliderTrack h="3px"><SliderFilledTrack /></SliderTrack>
                  <SliderThumb boxSize={3} />
                </Slider>
                <Text fontSize='xs' color='gray.400' minW="30px" textAlign="right">
                  {Math.round(bboxOpacity * 100)}%
                </Text>
              </Flex>

              {/* Slider épaisseur bordure */}
              <Flex align="center" gap={2} mt={2}>
                <Text fontSize='xs' color='gray.500' minW="50px">Bordure</Text>
                <Slider flex="1" value={bboxStroke} min={0.5} max={8} step={0.5}
                        onChange={(v) => setBboxStroke(v)}>
                  <SliderTrack h="3px"><SliderFilledTrack /></SliderTrack>
                  <SliderThumb boxSize={3} />
                </Slider>
                <Text fontSize='xs' color='gray.400' minW="30px" textAlign="right">
                  {bboxStroke}px
                </Text>
              </Flex>

              <Text fontSize='xs' color='gray.400' mt={3} lineHeight='1.4'>
                Clic droit sur une bbox pour la supprimer
              </Text>
            </Box>
          </Flex>
        </Center>
      </ThemeSwitcher>
    </ChakraProvider>
  )
}

export default withStreamlitConnection(Detection)
