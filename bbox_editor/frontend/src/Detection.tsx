import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps
} from "streamlit-component-lib"
import React, { useCallback, useEffect, useState } from "react"
import { ChakraProvider, SimpleGrid, Box, Flex, Center, Button, Text, Badge, Slider, SliderTrack, SliderFilledTrack, SliderThumb } from '@chakra-ui/react'

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
  token: number,
  image_name?: string,
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
    token,
    image_name = "",
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
  const [bboxOpacity,  setBboxOpacity]  = useState(1.0)
  const [bboxStroke,   setBboxStroke]   = useState(line_width)
  const [brightness,   setBrightness]   = useState(0)
  const [contrast,     setContrast]     = useState(0)
  const [highlightMode, setHighlightMode] = useState(false)

  // Nom de fichier court (sans chemin)
  const displayName = image_name ? image_name.replace(/.*[/\\]/, '') : ''

  const handleClassSelect = useCallback((l: string) => {
    setLabel(l)
    setRectangles(prev => {
      if (selectedId === null) return prev
      return prev.map(r => r.id === selectedId ? { ...r, label: l, stroke: color_map[l] } : r)
    })
  }, [selectedId, color_map])

  const [scale, setScale] = useState(1.0)

  useEffect(() => {
    const resizeCanvas = () => {
      const isMobile = window.innerWidth < 768
      // Sur desktop : on soustrait le panneau latéral fixe + le gap
      const reserved = isMobile ? 0 : SIDE_PANEL_W + GAP
      const availableWidth = Math.max(100, window.innerWidth - reserved)
      const scale_ratio = availableWidth / image_size[0]
      // Downscale si l'image dépasse la largeur dispo ; jamais d'upscaling
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

  const sendValue = useCallback(() => {
    Streamlit.setComponentValue({
      token,
      bboxes: rectangles.map((rect) => ({
        bbox: [rect.x, rect.y, rect.width, rect.height],
        label_id: label_list.indexOf(rect.label),
        label: rect.label
      }))
    })
  }, [rectangles, token, label_list])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      // Ne pas intercepter quand on est dans un champ texte
      const tag = (event.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      if (use_space && event.key === ' ') sendValue()
      if (event.key === 'Enter') sendValue()
      if (event.key === 'h' || event.key === 'H') setHighlightMode(prev => !prev)

      const num = parseInt(event.key)
      if (!isNaN(num) && num >= 1 && num <= label_list.length) {
        handleClassSelect(label_list[num - 1])
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => { window.removeEventListener('keydown', handleKeyDown) }
  }, [sendValue, handleClassSelect, label_list, use_space])

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
                brightness={brightness}
                contrast={contrast}
                fillOpacity={highlightMode ? 0.28 : 0}
              />
            </Box>

            {/* Panneau de contrôle — largeur fixe sur desktop */}
            <Box
              flexShrink={0}
              width={{ base: '100%', md: `${SIDE_PANEL_W}px` }}
              pl={{ md: 2 }}
            >
              {/* Nom de l'image courante */}
              {displayName && (
                <Box mb={3} p={2} borderRadius='md' borderLeft='3px solid' borderLeftColor='blue.400'
                     bg='blue.50' _dark={{ bg: 'blue.900', borderLeftColor: 'blue.300' }}>
                  <Text fontSize='9px' textTransform='uppercase' letterSpacing='wide' color='blue.500' mb='1px'>
                    Image en cours
                  </Text>
                  <Text fontSize='xs' fontWeight='bold' color='blue.700' _dark={{ color: 'blue.200' }}
                        overflow='hidden' textOverflow='ellipsis' whiteSpace='nowrap' title={image_name}>
                    {displayName}
                  </Text>
                </Box>
              )}

              {/* Classe + compteur bbox */}
              <Flex align="center" justify="space-between" mb={3}>
                <Text fontSize='sm' fontWeight='semibold' color='gray.600'>
                  Classe
                </Text>
                <Badge colorScheme='blue' fontSize='xs' px={2} borderRadius='full'>
                  {rectangles.length} bbox{rectangles.length !== 1 ? 's' : ''}
                </Badge>
              </Flex>

              {/* Boutons de classe + Valider côte à côte */}
              <Flex gap={3} align="stretch">
                <SimpleGrid columns={2} gap={2} flex="1">
                  {label_list.map((l, idx) => {
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
                        fontWeight={isActive ? 'bold' : 'normal'}
                        fontSize='xs'
                        justifyContent='flex-start'
                        px={2}
                      >
                        <Text as='span' fontSize='10px' fontWeight='bold' mr={1} opacity={0.7}>
                          {idx + 1}
                        </Text>
                        <Text as='span' overflow='hidden' textOverflow='ellipsis'>
                          {l}
                        </Text>
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

              {/* Bouton highlight */}
              <Button
                size='sm'
                mt={3}
                width='100%'
                colorScheme={highlightMode ? 'orange' : 'gray'}
                variant={highlightMode ? 'solid' : 'outline'}
                onClick={() => setHighlightMode(prev => !prev)}
              >
                {highlightMode ? '🔆 Highlight ON' : '🔅 Highlight'}
              </Button>

              {/* Slider opacité */}
              <Flex align="center" gap={2} mt={3}>
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

              {/* Slider luminosité */}
              <Flex align="center" gap={2} mt={2}>
                <Text fontSize='xs' color='gray.500' minW="50px">Lumière</Text>
                <Slider flex="1" value={brightness} min={-0.5} max={0.5} step={0.02}
                        onChange={(v) => setBrightness(v)}>
                  <SliderTrack h="3px"><SliderFilledTrack /></SliderTrack>
                  <SliderThumb boxSize={3} />
                </Slider>
                <Text fontSize='xs' color='gray.400' minW="30px" textAlign="right">
                  {brightness >= 0 ? '+' : ''}{Math.round(brightness * 100)}
                </Text>
              </Flex>

              {/* Slider contraste */}
              <Flex align="center" gap={2} mt={2}>
                <Text fontSize='xs' color='gray.500' minW="50px">Contraste</Text>
                <Slider flex="1" value={contrast} min={-50} max={50} step={2}
                        onChange={(v) => setContrast(v)}>
                  <SliderTrack h="3px"><SliderFilledTrack /></SliderTrack>
                  <SliderThumb boxSize={3} />
                </Slider>
                <Text fontSize='xs' color='gray.400' minW="30px" textAlign="right">
                  {contrast >= 0 ? '+' : ''}{contrast}
                </Text>
              </Flex>

              {/* Aide raccourcis */}
              <Box mt={3} p={2} borderRadius='md' bg='gray.50' _dark={{ bg: 'gray.700' }}>
                <Text fontSize='9px' textTransform='uppercase' letterSpacing='wide' color='gray.400' mb={1}>
                  Raccourcis
                </Text>
                <Text fontSize='xs' color='gray.500' lineHeight='1.6'>
                  <strong>1–{label_list.length}</strong> classe · <strong>Entrée</strong> valider · <strong>H</strong> highlight
                  <br />Clic <strong>molette</strong> : supprimer bbox · Clic <strong>droit</strong> : déplacer la vue
                </Text>
              </Box>
            </Box>
          </Flex>
        </Center>
      </ThemeSwitcher>
    </ChakraProvider>
  )
}

export default withStreamlitConnection(Detection)
